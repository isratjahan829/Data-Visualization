# %% [markdown]
# # AgriHGT on Agri-Vision4: full experiment notebook
#
# This notebook runs the complete AgriHGT pipeline described in the paper and saves
# every number the paper needs: leakage-controlled split, backbone adaptation and
# feature extraction, heterogeneous graph construction, HGT training (0/1/2 layers),
# the logistic-regression reference, the A0-A8 component ablation, the result tables,
# and a `paper_numbers.json` file with all dataset and graph counts.
#
# **Before running (Kaggle):**
# 1. Download Agri-Vision4 from Mendeley Data: https://data.mendeley.com/datasets/8t6k37ztxc/1
#    ("Download All"), then on Kaggle: *Datasets -> New Dataset -> upload the zip*.
#    Attach it to this notebook with *Add Data*. (If you skip this, the notebook tries to
#    download the zip itself, which needs Internet and may fail.)
# 2. *Settings -> Accelerator: GPU T4 x1* and *Settings -> Internet: On* (pretrained weights).
# 3. *Save Version -> Save & Run All (Commit)* so it runs in the background.
#
# Every stage caches its output under `/kaggle/working/agrihgt`, so if a session stops
# you can re-run and it continues where it stopped (attach the previous version's output
# as input and copy it back, or just re-run: fast stages recompute quickly).
# At the end, download `/kaggle/working/agrihgt_results.zip`.

# %%
%pip install -q "timm>=1.0.20" imagehash

# %%
# ---------------------------------------------------------------- configuration
import os, re, json, math, time, random, hashlib, zipfile, shutil, copy, warnings, subprocess, sys
import urllib.request
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import imagehash
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning)

TEST_MODE = os.environ.get("AGRIHGT_TEST") == "1"  # tiny synthetic run used only to check the code
ON_KAGGLE = Path("/kaggle").exists()
WORK = Path(os.environ.get("AGRIHGT_WORK", "/kaggle/working/agrihgt" if ON_KAGGLE else "./agrihgt_work"))
INPUT_ROOTS = [Path(p) for p in os.environ.get("AGRIHGT_INPUT", "/kaggle/input").split(":")]
MENDELEY_PAGE = "https://data.mendeley.com/datasets/8t6k37ztxc/1"
MENDELEY_ZIP_URL = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/8t6k37ztxc-1.zip"

SEED = 42
PHASH_SIZE = 16          # 256-bit perceptual hash
PHASH_THR = 6            # Hamming distance for "near duplicate"
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
SUP_FRACTION = 0.25      # train_sup share of the training partition
KNN_K = 5
CACHE_SIDE = 352         # images are cached with this shorter side (>= every backbone's crop)
N_PROC = max(1, min(4, os.cpu_count() or 1))

# Backbones, in the order they are run (DINOv3 first: it is also used for the ablation).
BACKBONE_ORDER = ["DINOv3", "DINOv2", "SwinTiny", "ConvNeXtTiny", "EfficientNetB3", "DenseNet121", "ResNet50"]
ADAPT = dict(epochs=15, patience=3, lr=1e-4, weight_decay=1e-4, batch=32, workers=N_PROC)
HGT_CFG = dict(hidden=256, heads=4, dropout=0.2, lr=1e-3, weight_decay=1e-4, eta_min=1e-6,
               clip=1.0, min_delta=1e-4, gate_bias=2.0, dec_dim=128, dec_dropout=0.1)
EPOCH_RULE = {"ResNet50": (300, 300), "SwinTiny": (300, 300), "DenseNet121": (300, 300)}  # (max, patience)
DEFAULT_EPOCH_RULE = (400, 50)
DEPTHS = [0, 1, 2]
RUN_LOGREG = True
RUN_ABLATION = True
ABLATION_BACKBONE = "DINOv3"
LOGREG_MAX_ITER = 2000
PRETRAINED = True

if TEST_MODE:
    BACKBONE_ORDER = ["ResNet50"]
    ABLATION_BACKBONE = "ResNet50"
    ADAPT.update(epochs=1, patience=1, batch=8, workers=0)
    EPOCH_RULE = {"ResNet50": (6, 6)}
    DEFAULT_EPOCH_RULE = (6, 6)
    LOGREG_MAX_ITER = 200
    PRETRAINED = False
    CACHE_SIDE = 96
    N_PROC = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for sub in ["cache", "features", "predictions", "tables", "raw"]:
    (WORK / sub).mkdir(parents=True, exist_ok=True)
print("device:", DEVICE, "| work dir:", WORK, "| test mode:", TEST_MODE)

NUMBERS_PATH = WORK / "paper_numbers.json"
NUM = json.loads(NUMBERS_PATH.read_text()) if NUMBERS_PATH.exists() else {}


def save_numbers():
    NUMBERS_PATH.write_text(json.dumps(NUM, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything()

# %%
# ---------------------------------------------------------------- 1. locate the dataset
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RAW = WORK / "raw"


def list_images(roots):
    out = []
    for r in roots:
        if r.exists():
            out += [p for p in r.rglob("*") if p.suffix.lower() in IMG_EXT and p.is_file()]
    return sorted(set(out))


def extract_all_zips(roots, dest):
    """Extract every zip under roots into dest, repeating for zips found inside zips."""
    done = set()
    while True:
        zips = [z for r in list(roots) + [dest] if r.exists() for z in r.rglob("*.zip") if z not in done]
        if not zips:
            return
        for z in zips:
            done.add(z)
            target = dest / z.stem
            if not target.exists():
                print("extracting", z)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(target)


images = list_images(INPUT_ROOTS + [RAW])
if len(images) < 100:
    extract_all_zips(INPUT_ROOTS, RAW)
    images = list_images(INPUT_ROOTS + [RAW])
if len(images) < 100 and not TEST_MODE:
    print("No attached dataset found; trying to download it from Mendeley ...")
    try:
        zpath = WORK / "agrivision4.zip"
        urllib.request.urlretrieve(MENDELEY_ZIP_URL, zpath)
        extract_all_zips([WORK], RAW)
        images = list_images([RAW])
    except Exception as e:
        raise SystemExit(f"Download failed ({e}). Download 'Download All' from {MENDELEY_PAGE}, "
                         "upload it as a Kaggle dataset, attach it with 'Add Data', and run again.")
print("image files found:", len(images))

# %%
# ---------------------------------------------------------------- 2. parse crop, class, original/augmented
CROPS = {"bottlegourd": "Bottle Gourd", "papaya": "Papaya", "tomato": "Tomato", "zucchini": "Zucchini"}
# Folder-name spelling fixes so original and augmented folders map to one class.
CLASS_ALIASES = {"downymidew": "downymildew"}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def clean_class(folder, crop):
    name = re.sub(r"[_\-]+", " ", folder)
    for word in crop.split():
        name = re.sub(rf"\b{word}\b", " ", name, flags=re.I)
    name = re.sub(r"\s+", " ", name).strip(" .")
    key = CLASS_ALIASES.get(norm(name), norm(name))
    return name, key


def parse_path(p):
    parts = [norm(x) for x in p.parts[:-1]]
    crop = next((v for k, v in CROPS.items() if any(k in x for x in parts)), None)
    if crop is None:
        return None
    folder = p.parent.name
    if norm(folder) in CROPS:  # image sits directly in a crop folder: no class
        return None
    name, key = clean_class(folder, crop)
    joined = "/".join(parts)
    if "augment" in joined:
        aug = True
    elif "original" in joined:
        aug = False
    else:
        aug = "aug" in p.stem.lower()
    return crop, name, key, aug


rows, skipped = [], 0
for p in images:
    r = parse_path(p)
    if r is None:
        skipped += 1
        continue
    rows.append(dict(path=str(p), crop=r[0], cls_name=r[1], cls_key=r[2], is_aug=r[3]))
df = pd.DataFrame(rows)
# One display name per (crop, class key): prefer the name used by original images.
disp = (df.sort_values("is_aug").groupby(["crop", "cls_key"])["cls_name"].first())
df["cls_name"] = [disp[(c, k)] for c, k in zip(df.crop, df.cls_key)]
df["label_name"] = df.crop + " | " + df.cls_name

summary = df.groupby(["crop", "cls_name"]).is_aug.agg(original=lambda s: int((~s).sum()), augmented=lambda s: int(s.sum()))
pd.set_option("display.max_rows", 100)
print(summary)
print("skipped (no crop/class in path):", skipped)
bad = summary[(summary.original == 0)]
if len(bad):
    print("WARNING: classes without original images (check folder names / CLASS_ALIASES):\n", bad)
CLASSES = sorted(df.label_name.unique())
N_CLASSES = len(CLASSES)
print("classes:", N_CLASSES, "(expected 28)")
NUM.update(files_found=len(df), originals_found=int((~df.is_aug).sum()),
           classes=N_CLASSES, original_per_class={f"{c} | {k}": int(v) for (c, k), v in summary.original.items()})
save_numbers()

# %%
# ---------------------------------------------------------------- 3. exact-duplicate removal (MD5)
def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


with Pool(N_PROC) as pool:
    df["md5"] = pool.map(md5_file, df.path.tolist(), chunksize=64)
df = df.sort_values(["is_aug", "path"]).reset_index(drop=True)  # keep an original when one exists
cross_class = int((df.groupby("md5").label_name.nunique() > 1).sum())
n_before = len(df)
df = df.drop_duplicates("md5", keep="first").reset_index(drop=True)
NUM.update(md5_duplicates_removed=n_before - len(df), md5_cross_class_groups=cross_class,
           images_after_dedup=len(df), originals_after_dedup=int((~df.is_aug).sum()),
           augmented_after_dedup=int(df.is_aug.sum()))
save_numbers()
print({k: NUM[k] for k in ["md5_duplicates_removed", "images_after_dedup", "originals_after_dedup", "augmented_after_dedup"]})

# %%
# ---------------------------------------------------------------- 4. perceptual hashes + resized image cache
CACHE = WORK / "cache"


def phash_and_cache(args):
    src, dst = args
    im = Image.open(src)
    im.draft("RGB", (1024, 1024))  # fast JPEG decode at reduced scale; pHash works on a 64x64 resize anyway
    im = im.convert("RGB")
    bits = imagehash.phash(im, hash_size=PHASH_SIZE).hash.flatten()
    w, h = im.size
    s = CACHE_SIDE / min(w, h)
    if s < 1:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    if not Path(dst).exists():
        im.save(dst, quality=95)
    return np.packbits(bits)


df["cache"] = [str(CACHE / f"{i:06d}_{m[:8]}.jpg") for i, m in enumerate(df.md5)]
hash_file = WORK / "phash.npy"
if hash_file.exists() and len(np.load(hash_file)) == len(df):
    H = np.load(hash_file)
else:
    t0 = time.time()
    with Pool(N_PROC) as pool:
        H = np.stack(pool.map(phash_and_cache, list(zip(df.path, df.cache)), chunksize=32))
    np.save(hash_file, H)
    print(f"hashed and cached {len(df)} images in {time.time() - t0:.0f}s")

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming(A, B, chunk=256):
    out = np.empty((len(A), len(B)), dtype=np.uint16)
    for i in range(0, len(A), chunk):
        out[i:i + chunk] = POP[A[i:i + chunk, None, :] ^ B[None, :, :]].sum(-1, dtype=np.uint16)
    return out

# %%
# ---------------------------------------------------------------- 5. duplicate-aware split, audit, support control
label_to_idx = {c: i for i, c in enumerate(CLASSES)}
df["y"] = df.label_name.map(label_to_idx)
orig_idx = df.index[~df.is_aug].to_numpy()
tr, tmp = train_test_split(orig_idx, test_size=1 - SPLIT_FRACTIONS[0], stratify=df.y[orig_idx], random_state=SEED)
va, te = train_test_split(tmp, test_size=SPLIT_FRACTIONS[2] / (1 - SPLIT_FRACTIONS[0]),
                          stratify=df.y[tmp], random_state=SEED)
part = np.array(["train"] * len(df), dtype=object)
part[va], part[te] = "val", "test"
group = np.arange(len(df))
NUM.update(initial_split=dict(train=len(tr), val=len(va), test=len(te)))

# link each augmented image to its closest same-class original (Hamming <= threshold)
linked = 0
for y in range(N_CLASSES):
    A = df.index[(df.y == y) & df.is_aug].to_numpy()
    O = df.index[(df.y == y) & ~df.is_aug].to_numpy()
    if len(A) == 0 or len(O) == 0:
        continue
    D = hamming(H[A], H[O])
    j, d = D.argmin(1), D.min(1)
    ok = d <= PHASH_THR
    part[A[ok]] = part[O[j[ok]]]
    group[A[ok]] = O[j[ok]]
    linked += int(ok.sum())
NUM.update(augmented_linked=linked, augmented_unlinked=int(df.is_aug.sum()) - linked)

# same-class near-duplicate structure (hashes never change, so compute once)
class_idx = {y: df.index[df.y == y].to_numpy() for y in range(N_CLASSES)}
close = {}
for y, idx in class_idx.items():
    c = hamming(H[idx], H[idx]) <= PHASH_THR
    np.fill_diagonal(c, False)
    close[y] = c


def audit():
    rounds = []
    while True:
        moved, n_viol = set(), 0
        for y, idx in class_idx.items():
            P = part[idx]
            cross = np.triu(close[y] & (P[:, None] != P[None, :]), 1)
            ii, jj = np.nonzero(cross)
            n_viol += len(ii)
            for a, b in zip(ii, jj):
                pa, pb = P[a], P[b]
                mv = b if pa == "train" else a if pb == "train" else (a if pa == "test" else b)
                moved.add(idx[mv])
        rounds.append(n_viol)
        if n_viol == 0:
            return rounds
        part[list(moved)] = "train"


rounds = audit()
print("audit rounds (violating pairs):", rounds)

# per-class support control
rng = np.random.default_rng(SEED)
support_moves = 0
for y, idx in class_idx.items():
    has_dup = close[y].any(1)
    for need in ["val", "test"]:
        if (part[idx] == need).any():
            continue
        cands = [k for k in np.argsort(df.is_aug.values[idx], kind="stable") if part[idx[k]] == "train"]
        rng.shuffle(cands)
        cands = sorted(cands, key=lambda k: df.is_aug.values[idx[k]])  # originals first
        for k in cands:
            if not has_dup[k]:  # moving it creates no cross-partition near duplicate
                part[idx[k]] = need
                support_moves += 1
                break
        else:
            print(f"WARNING: no eligible image to give class {CLASSES[y]} a {need} sample")
final_rounds = audit()
df["part"] = part
df["group"] = group
counts = df.part.value_counts().to_dict()
NUM.update(audit_rounds=rounds, support_moves=support_moves, audit_after_support=final_rounds,
           final_split=dict(train=counts.get("train", 0), val=counts.get("val", 0), test=counts.get("test", 0)),
           min_val_per_class=int(df[df.part == "val"].y.value_counts().reindex(range(N_CLASSES), fill_value=0).min()),
           min_test_per_class=int(df[df.part == "test"].y.value_counts().reindex(range(N_CLASSES), fill_value=0).min()))
print(NUM["final_split"], "| min val/test per class:", NUM["min_val_per_class"], NUM["min_test_per_class"])

# %%
# ---------------------------------------------------------------- 6. train_msg / train_sup partition
tr_idx = df.index[df.part == "train"].to_numpy()
sgkf = StratifiedGroupKFold(n_splits=round(1 / SUP_FRACTION), shuffle=True, random_state=SEED)
_, sup_pos = next(sgkf.split(tr_idx, df.y[tr_idx], df.group[tr_idx]))
df["role"] = df.part
df.loc[tr_idx, "role"] = "train_msg"
df.loc[tr_idx[sup_pos], "role"] = "train_sup"
NUM.update(train_msg=int((df.role == "train_msg").sum()), train_sup=int((df.role == "train_sup").sum()))
df.drop(columns=["md5"]).to_csv(WORK / "split.csv", index=False)
save_numbers()
print({k: NUM[k] for k in ["train_msg", "train_sup"]})

# %%
# ---------------------------------------------------------------- 7. backbones: partial fine-tuning + feature extraction
IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class HeadWrap(nn.Module):
    """Backbone that returns a feature vector, followed by a temporary classification head."""

    def __init__(self, backbone, dim, n_classes):
        super().__init__()
        self.backbone, self.head = backbone, nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def build_backbone(key, n_classes, pretrained=True):
    """Returns (model, trainable modules, feature dim, input size, mean, std, source note)."""
    tv = torchvision.models
    if key == "ResNet50":
        m = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(2048, n_classes)
        return m, [m.layer4, m.fc], 2048, 224, *IMNET, "torchvision"
    if key == "DenseNet121":
        m = tv.densenet121(weights=tv.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Linear(1024, n_classes)
        return m, [m.features.denseblock4, m.features.norm5, m.classifier], 1024, 224, *IMNET, "torchvision"
    if key == "EfficientNetB3":
        m = tv.efficientnet_b3(weights=tv.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(1536, n_classes)
        return m, [m.features[7:], m.classifier], 1536, 300, *IMNET, "torchvision"
    if key == "ConvNeXtTiny":
        m = tv.convnext_tiny(weights=tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[2] = nn.Linear(768, n_classes)
        return m, [m.features[7], m.classifier], 768, 224, *IMNET, "torchvision"
    if key == "SwinTiny":
        m = tv.swin_t(weights=tv.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        m.head = nn.Linear(768, n_classes)
        return m, [m.features[7], m.norm, m.head], 768, 224, *IMNET, "torchvision"
    if key == "DINOv2":
        try:
            bb = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=pretrained)
            src = "official DINOv2 (torch.hub)"
        except Exception as e:  # fall back to the timm port of the same weights
            print("torch.hub DINOv2 failed, using timm:", e)
            import timm
            bb = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=pretrained, num_classes=0, img_size=224)
            src = "timm vit_base_patch14_dinov2.lvd142m"
        m = HeadWrap(bb, 768, n_classes)
        return m, [bb.blocks[-2:], bb.norm, m.head], 768, 224, *IMNET, src
    if key == "DINOv3":
        import timm
        bb = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=pretrained, num_classes=0)
        cfg = bb.pretrained_cfg
        m = HeadWrap(bb, bb.num_features, n_classes)
        mods = [bb.blocks[-2:], bb.norm] + ([bb.fc_norm] if hasattr(bb, "fc_norm") else [])
        return m, mods + [m.head], bb.num_features, cfg["input_size"][-1], cfg["mean"], cfg["std"], "timm vit_base_patch16_dinov3.lvd1689m"
    raise ValueError(key)


def drop_head(key, m):
    if key == "ResNet50":
        m.fc = nn.Identity()
    elif key == "DenseNet121":
        m.classifier = nn.Identity()
    elif key == "EfficientNetB3":
        m.classifier = nn.Identity()
    elif key == "ConvNeXtTiny":
        m.classifier[2] = nn.Identity()
    elif key == "SwinTiny":
        m.head = nn.Identity()
    else:
        m.head = nn.Identity()
    return m


class ImgDS(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = list(paths), list(labels), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def transforms_for(size, mean, std, train):
    if train:
        return T.Compose([T.RandomResizedCrop(size, scale=(0.7, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                          T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(), T.Normalize(mean, std)])
    return T.Compose([T.Resize(int(round(size / 0.875))), T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])


def set_train_mode(model, train_mods):
    model.eval()  # frozen parts (incl. BatchNorm statistics) stay fixed
    for mod in train_mods:
        mod.train()


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            p = model(x.to(DEVICE)).argmax(1).cpu()
        correct += int((p == y).sum())
        total += len(y)
    return correct / max(total, 1)


def adapt_and_extract(key):
    feat_path = WORK / "features" / f"{key}.npy"
    meta_path = WORK / "features" / f"{key}.json"
    if feat_path.exists() and meta_path.exists():
        return np.load(feat_path), json.loads(meta_path.read_text())
    seed_everything()
    model, train_mods, dim, size, mean, std, src = build_backbone(key, N_CLASSES, PRETRAINED)
    for p_ in model.parameters():
        p_.requires_grad = False
    for mod in train_mods:
        for p_ in mod.parameters():
            p_.requires_grad = True
    model.to(DEVICE)
    trn, val = df[df.part == "train"], df[df.part == "val"]
    kw = dict(num_workers=ADAPT["workers"], pin_memory=DEVICE.type == "cuda")
    dl_tr = DataLoader(ImgDS(trn.cache, trn.y, transforms_for(size, mean, std, True)), batch_size=ADAPT["batch"], shuffle=True, drop_last=True, **kw)
    dl_va = DataLoader(ImgDS(val.cache, val.y, transforms_for(size, mean, std, False)), batch_size=ADAPT["batch"] * 2, **kw)
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=ADAPT["lr"], weight_decay=ADAPT["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ADAPT["epochs"])
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    best, best_state, bad, t0, hist = -1.0, None, 0, time.time(), []
    for ep in range(ADAPT["epochs"]):
        set_train_mode(model, train_mods)
        for x, y in dl_tr:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                loss = F.cross_entropy(model(x.to(DEVICE)), y.to(DEVICE))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        acc = accuracy(model, dl_va)
        hist.append(acc)
        print(f"  [{key}] adapt epoch {ep + 1}: val acc {acc:.4f}")
        if acc > best + 1e-4:
            best, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= ADAPT["patience"]:
                break
    model.load_state_dict(best_state)
    model = drop_head(key, model).eval()
    for p_ in model.parameters():
        p_.requires_grad = False
    dl_all = DataLoader(ImgDS(df.cache, df.y, transforms_for(size, mean, std, False)), batch_size=ADAPT["batch"] * 2, **kw)
    feats = []
    with torch.no_grad():
        for x, _ in dl_all:
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                feats.append(model(x.to(DEVICE)).float().cpu())
    X = torch.cat(feats).numpy().astype(np.float32)
    meta = dict(source=src, dim=int(X.shape[1]), input_size=int(size), adapt_best_val_acc=best,
                adapt_epochs_run=len(hist), adapt_minutes=round((time.time() - t0) / 60, 1),
                trainable_params=int(sum(p_.numel() for p_ in params)))
    np.save(feat_path, X)
    meta_path.write_text(json.dumps(meta, indent=2))
    del model
    torch.cuda.empty_cache()
    return X, meta

# %%
# ---------------------------------------------------------------- 8. heterogeneous graph
CATEGORY = {  # scientific choice; verify with an agronomist / supervisor
    "Bottle Gourd": {"alternarialeafblight": "Fungal", "anthracnose": "Fungal", "downymildew": "Fungal",
                     "earlyalternarialeafblight": "Fungal", "fungaldamageleaf": "Fungal", "healthy": "Healthy",
                     "mosaicvirus": "Viral"},
    "Papaya": {"bacterialblight": "Bacterial", "caricainsecthole": "Pest", "curledyellowspot": "Viral",
               "healthyleaf": "Healthy", "healthy": "Healthy", "mosaicvirus": "Viral", "pathogensymptoms": "Other",
               "yellownecroticspotsholes": "Other"},
    "Tomato": {"downy": "Fungal", "healthy": "Healthy", "mosaic": "Viral", "spot": "Other", "whitespot": "Pest"},
    "Zucchini": {"angularleafspot": "Bacterial", "anthracnose": "Fungal", "downymildew": "Fungal", "dryleaf": "Other",
                 "healthy": "Healthy", "insectdamage": "Pest", "ironchlorosisdamage": "Nutrient Deficiency",
                 "xanthomonasleafspot": "Bacterial", "yellowmosaicvirus": "Viral"},
}


def category_of(label_name):
    crop, cls = label_name.split(" | ")
    key = CLASS_ALIASES.get(norm(cls), norm(cls))
    if key in CATEGORY.get(crop, {}):
        return CATEGORY[crop][key]
    for kw, cat in [("healthy", "Healthy"), ("virus", "Viral"), ("mosaic", "Viral"), ("curl", "Viral"),
                    ("bacteri", "Bacterial"), ("xanthomonas", "Bacterial"), ("insect", "Pest"), ("pest", "Pest"),
                    ("iron", "Nutrient Deficiency"), ("deficien", "Nutrient Deficiency"), ("fung", "Fungal"),
                    ("mildew", "Fungal"), ("anthracnose", "Fungal"), ("alternaria", "Fungal"), ("blight", "Fungal")]:
        if kw in key:
            return cat
    return "Other"


CROP_NAMES = sorted(df.crop.unique())
CAT_NAMES = sorted({category_of(c) for c in CLASSES})
disease_crop = np.array([CROP_NAMES.index(c.split(" | ")[0]) for c in CLASSES])
disease_cat = np.array([CAT_NAMES.index(category_of(c)) for c in CLASSES])
leaf_crop = df.crop.map({c: i for i, c in enumerate(CROP_NAMES)}).to_numpy()
print(pd.DataFrame(dict(disease=CLASSES, category=[CAT_NAMES[i] for i in disease_cat])).to_string())
NUM.update(disease_categories={c: CAT_NAMES[k] for c, k in zip(CLASSES, disease_cat)})


def knn_within_crop(X, k=KNN_K):
    Xn = F.normalize(torch.from_numpy(X).to(DEVICE), dim=1)
    src, dst = [], []
    for c in np.unique(leaf_crop):
        idx = torch.from_numpy(np.nonzero(leaf_crop == c)[0]).to(DEVICE)
        S = Xn[idx] @ Xn[idx].T  # cosine similarity = 1 - cosine distance
        S.fill_diagonal_(-float("inf"))
        nb = S.topk(min(k, len(idx) - 1), dim=1).indices
        src.append(idx.repeat_interleave(nb.shape[1]))
        dst.append(idx[nb.reshape(-1)])
    return torch.cat(src).cpu(), torch.cat(dst).cpu()


def build_graph(X, drop=()):
    """drop may contain: 'similar', 'crop', 'category', 'has_disease'."""
    N = len(X)
    n = {"leaf": N, "disease": N_CLASSES}
    if "crop" not in drop:
        n["crop"] = len(CROP_NAMES)
    if "category" not in drop:
        n["category"] = len(CAT_NAMES)
    t = torch.as_tensor
    E = {}
    if "crop" not in drop:
        E[("leaf", "grown_on", "crop")] = (t(np.arange(N)), t(leaf_crop))
        E[("crop", "affects", "disease")] = (t(disease_crop), t(np.arange(N_CLASSES)))
    if "category" not in drop:
        E[("disease", "is_type_of", "category")] = (t(np.arange(N_CLASSES)), t(disease_cat))
        a, b = np.nonzero((disease_cat[:, None] == disease_cat[None, :]) & ~np.eye(N_CLASSES, dtype=bool))
        E[("disease", "co_category", "disease")] = (t(a), t(b))
    if "similar" not in drop:
        E[("leaf", "similar_to", "leaf")] = knn_within_crop(X)
    if "has_disease" not in drop:
        msg = np.nonzero((df.role == "train_msg").to_numpy())[0]
        E[("leaf", "has_disease", "disease")] = (t(msg), t(df.y.to_numpy()[msg]))
    for (s, r, d), (a, b) in list(E.items()):
        E[(d, "rev_" + r, s)] = (b, a)
    for nt, cnt in n.items():
        E[(nt, "self", nt)] = (t(np.arange(cnt)), t(np.arange(cnt)))
    E = {k: (a.long().to(DEVICE), b.long().to(DEVICE)) for k, (a, b) in E.items()}
    return n, E


def graph_counts(n, E):
    fwd = {r: int(a.numel()) for (s, r, d), (a, b) in E.items() if not r.startswith("rev_") and r != "self"}
    return dict(nodes=n, total_nodes=sum(n.values()), forward_edges=fwd,
                self_edges=sum(int(a.numel()) for (s, r, d), (a, b) in E.items() if r == "self"),
                relation_types=len({r for (_, r, _) in E}), total_edges=sum(int(a.numel()) for a, _ in E.values()))

# %%
# ---------------------------------------------------------------- 9. AgriHGT model (HGT in plain PyTorch)
class HGTLayer(nn.Module):
    """Heterogeneous Graph Transformer layer (Hu et al., WWW 2020): node-type-specific K/Q/V/A,
    relation-specific attention and message matrices with a learnable prior, softmax over all
    incoming edges of a target node, and a gated skip connection with LayerNorm."""

    def __init__(self, node_types, relations, dim, heads, dropout):
        super().__init__()
        self.h, self.dk = heads, dim // heads
        L = lambda: nn.ModuleDict({t: nn.Linear(dim, dim) for t in node_types})
        self.k, self.q, self.v, self.a = L(), L(), L(), L()
        self.norm = nn.ModuleDict({t: nn.LayerNorm(dim) for t in node_types})
        self.skip = nn.ParameterDict({t: nn.Parameter(torch.ones(1)) for t in node_types})
        self.rel_att = nn.ParameterDict({r: nn.Parameter(torch.eye(self.dk).repeat(heads, 1, 1)) for r in relations})
        self.rel_msg = nn.ParameterDict({r: nn.Parameter(torch.eye(self.dk).repeat(heads, 1, 1)) for r in relations})
        self.rel_pri = nn.ParameterDict({r: nn.Parameter(torch.ones(heads)) for r in relations})
        self.drop = nn.Dropout(dropout)

    def forward(self, h, E):
        scores, msgs, dsts = defaultdict(list), defaultdict(list), defaultdict(list)
        K = {t: self.k[t](x).view(-1, self.h, self.dk) for t, x in h.items()}
        Q = {t: self.q[t](x).view(-1, self.h, self.dk) for t, x in h.items()}
        V = {t: self.v[t](x).view(-1, self.h, self.dk) for t, x in h.items()}
        for (s, r, d), (src, dst) in E.items():
            k = torch.einsum("ehd,hdf->ehf", K[s][src], self.rel_att[r])
            scores[d].append((Q[d][dst] * k).sum(-1) * self.rel_pri[r] / math.sqrt(self.dk))
            msgs[d].append(torch.einsum("ehd,hdf->ehf", V[s][src], self.rel_msg[r]))
            dsts[d].append(dst)
        out = {}
        for t, x in h.items():
            if t not in scores:
                out[t] = x
                continue
            sc, mg, ds = torch.cat(scores[t]), torch.cat(msgs[t]), torch.cat(dsts[t])
            n = x.shape[0]
            mx = torch.full((n, self.h), -float("inf"), device=x.device).scatter_reduce(
                0, ds[:, None].expand(-1, self.h), sc, reduce="amax", include_self=True)
            e = torch.exp(sc - mx[ds])
            den = torch.zeros(n, self.h, device=x.device).index_add_(0, ds, e)
            att = e / den[ds].clamp_min(1e-16)
            agg = torch.zeros(n, self.h, self.dk, device=x.device).index_add_(0, ds, att.unsqueeze(-1) * mg)
            trans = self.drop(self.a[t](F.gelu(agg.reshape(n, -1))))
            g = torch.sigmoid(self.skip[t])
            out[t] = self.norm[t](g * trans + (1 - g) * x)
        return out


class AgriHGT(nn.Module):
    def __init__(self, feat_dim, n_nodes, relations, n_layers, cfg=HGT_CFG, zero_meta=False, residual=True, gate="learned"):
        super().__init__()
        H = cfg["hidden"]
        self.types = list(n_nodes)
        self.proj = nn.ModuleDict({t: nn.Linear(feat_dim if t == "leaf" else H, H) for t in self.types})
        emb = {t: torch.randn(c, H) * 0.02 for t, c in n_nodes.items() if t != "leaf"}
        if zero_meta:  # A6: fixed zero inputs for crop / disease / category nodes
            for t, e in emb.items():
                self.register_buffer(f"emb_{t}", torch.zeros_like(e))
        else:
            for t, e in emb.items():
                self.register_parameter(f"emb_{t}", nn.Parameter(e))
        self.layers = nn.ModuleList([HGTLayer(self.types, relations, H, cfg["heads"], cfg["dropout"]) for _ in range(n_layers)])
        self.residual, self.gate_mode = residual, gate
        self.res = nn.Sequential(nn.Linear(feat_dim, H), nn.LayerNorm(H))
        self.gate = nn.Linear(2 * H, H)
        nn.init.constant_(self.gate.bias, cfg["gate_bias"])
        self.dec_leaf = nn.Linear(H, cfg["dec_dim"])
        self.dec_dis = nn.Linear(H, cfg["dec_dim"], bias=False)
        self.dec_drop = nn.Dropout(cfg["dec_dropout"])
        self.dec_out = nn.Linear(cfg["dec_dim"], 1)

    def forward(self, x_leaf, E, idx):
        h = {t: torch.tanh(self.proj[t](x_leaf if t == "leaf" else getattr(self, f"emb_{t}"))) for t in self.types}
        for layer in self.layers:
            h = layer(h, E)
        hl = h["leaf"][idx]
        if not self.residual:  # A7
            z = hl
        else:
            r = self.res(x_leaf[idx])
            if self.gate_mode == "learned":
                g = torch.sigmoid(self.gate(torch.cat([r, hl], -1)))
                z = g * r + (1 - g) * hl
            else:  # A8: fixed equal-weight average
                z = 0.5 * r + 0.5 * hl
        u = self.dec_leaf(z)[:, None, :]
        v = self.dec_dis(h["disease"])[None, :, :]
        return self.dec_out(self.dec_drop(F.relu(u + v))).squeeze(-1)  # (len(idx), n_classes)

# %%
# ---------------------------------------------------------------- 10. training, logistic regression, saving predictions
Y = torch.as_tensor(df.y.to_numpy(), device=DEVICE)
IDX = {r: torch.as_tensor(np.nonzero((df.role == r).to_numpy())[0], device=DEVICE) for r in ["train_sup", "val", "test"]}
PRED_DIR = WORK / "predictions"
RUNS_PATH = WORK / "runs.csv"
RUNS = pd.read_csv(RUNS_PATH).to_dict("records") if RUNS_PATH.exists() else []


def record(run_id, **kw):
    global RUNS
    RUNS = [r for r in RUNS if r["run_id"] != run_id] + [dict(run_id=run_id, **kw)]
    pd.DataFrame(RUNS).to_csv(RUNS_PATH, index=False)


def save_pred(run_id, probs_test, val_acc):
    y_true = df.y.to_numpy()[df.role.to_numpy() == "test"]
    np.savez(PRED_DIR / f"{run_id}.npz", y_true=y_true, y_pred=probs_test.argmax(1), probs=probs_test,
             val_acc=val_acc, class_names=np.array(CLASSES))
    acc = float((probs_test.argmax(1) == y_true).mean())
    f1 = float(f1_score(y_true, probs_test.argmax(1), average="macro"))
    return acc, f1


def train_hgt(run_id, X, graph, n_layers, max_epochs, patience, **variant):
    if (PRED_DIR / f"{run_id}.npz").exists():
        print("skip (done):", run_id)
        return
    seed_everything()
    n_nodes, E = graph
    relations = sorted({r for (_, r, _) in E})
    x = torch.as_tensor(X, device=DEVICE)
    model = AgriHGT(X.shape[1], n_nodes, relations, n_layers, **variant).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=HGT_CFG["lr"], weight_decay=HGT_CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=HGT_CFG["eta_min"])
    y_sup = Y[IDX["train_sup"]]
    cnt = torch.bincount(y_sup, minlength=N_CLASSES).float().clamp_min(1)
    w = 1.0 / cnt
    w = w / w.mean()
    best, best_state, bad, t0, ep = -1.0, None, 0, time.time(), 0
    for ep in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x, E, IDX["train_sup"]), y_sup, weight=w)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), HGT_CFG["clip"])
        opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            va = (model(x, E, IDX["val"]).argmax(1) == Y[IDX["val"]]).float().mean().item()
        if va > best + HGT_CFG["min_delta"]:
            best, bad = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(x, E, IDX["test"]), -1).cpu().numpy()
    acc, f1 = save_pred(run_id, probs, best)
    record(run_id, model="AgriHGT", layers=n_layers, val_acc=best, test_acc=acc, test_macro_f1=f1,
           epochs=ep, minutes=round((time.time() - t0) / 60, 2))
    print(f"{run_id}: val {best:.4f} | test acc {acc:.4f} | macro-F1 {f1:.4f} | {ep} epochs")


def run_logreg(run_id, X):
    if (PRED_DIR / f"{run_id}.npz").exists():
        print("skip (done):", run_id)
        return
    t0 = time.time()
    tr = (df.part == "train").to_numpy()
    clf = LogisticRegression(max_iter=LOGREG_MAX_ITER)
    clf.fit(X[tr], df.y.to_numpy()[tr])
    va = (df.part == "val").to_numpy()
    val_acc = float((clf.predict(X[va]) == df.y.to_numpy()[va]).mean())
    te = (df.role == "test").to_numpy()
    probs = np.zeros((int(te.sum()), N_CLASSES))
    probs[:, clf.classes_] = clf.predict_proba(X[te])
    acc, f1 = save_pred(run_id, probs, val_acc)
    record(run_id, model="LogReg", layers=-1, val_acc=val_acc, test_acc=acc, test_macro_f1=f1,
           epochs=int(np.max(clf.n_iter_)), minutes=round((time.time() - t0) / 60, 2))
    print(f"{run_id}: val {val_acc:.4f} | test acc {acc:.4f} | macro-F1 {f1:.4f}")

# %%
# ---------------------------------------------------------------- 11. main experiments: 7 backbones x (LR, HGT-0/1/2)
NUM.setdefault("backbones", {})
for key in BACKBONE_ORDER:
    print(f"\n===== {key} =====")
    Xraw, meta = adapt_and_extract(key)
    tr = (df.part == "train").to_numpy()
    mu, sd = Xraw[tr].mean(0), Xraw[tr].std(0)
    X = ((Xraw - mu) / np.where(sd < 1e-6, 1.0, sd)).astype(np.float32)
    graph = build_graph(X)
    meta["graph"] = graph_counts(*graph)
    NUM["backbones"][key] = meta
    save_numbers()
    if RUN_LOGREG:
        run_logreg(f"{key}_LR", X)
    max_ep, pat = EPOCH_RULE.get(key, DEFAULT_EPOCH_RULE)
    for L in DEPTHS:
        train_hgt(f"{key}_L{L}", X, graph, L, max_ep, pat)
    del graph
    torch.cuda.empty_cache()

# %%
# ---------------------------------------------------------------- 12. component ablation A0-A8
if RUN_ABLATION:
    key = ABLATION_BACKBONE
    runs = pd.DataFrame(RUNS)
    depth_runs = runs[runs.run_id.isin([f"{key}_L{L}" for L in DEPTHS])]
    best_L = int(depth_runs.sort_values(["val_acc", "layers"], ascending=[False, False]).iloc[0].layers)
    NUM["ablation_depth_selected_by_val"] = best_L
    print("ablation reference depth (best validation accuracy):", best_L)
    Xraw, _ = adapt_and_extract(key)
    tr = (df.part == "train").to_numpy()
    mu, sd = Xraw[tr].mean(0), Xraw[tr].std(0)
    X = ((Xraw - mu) / np.where(sd < 1e-6, 1.0, sd)).astype(np.float32)
    max_ep, pat = DEFAULT_EPOCH_RULE
    # A0 = full model at the selected depth; A1 = zero HGT layers (same settings): reuse those runs.
    for a_id, src in [("A0", f"{key}_L{best_L}"), ("A1", f"{key}_L0")]:
        shutil.copy(PRED_DIR / f"{src}.npz", PRED_DIR / f"{a_id}.npz")
        r = next(r for r in RUNS if r["run_id"] == src)
        record(a_id, **{k: v for k, v in r.items() if k != "run_id"})
    full = build_graph(X)
    for a_id, drop in [("A2", ("similar",)), ("A3", ("crop",)), ("A4", ("category",)), ("A5", ("has_disease",))]:
        g = build_graph(X, drop)
        NUM.setdefault("ablation_graphs", {})[a_id] = graph_counts(*g)
        train_hgt(a_id, X, g, best_L, max_ep, pat)
    train_hgt("A6", X, full, best_L, max_ep, pat, zero_meta=True)
    train_hgt("A7", X, full, best_L, max_ep, pat, residual=False)
    train_hgt("A8", X, full, best_L, max_ep, pat, gate="fixed")
    save_numbers()

# %%
# ---------------------------------------------------------------- 13. result tables for the paper (LaTeX rows)
METRICS_SCRIPT = r'''<<AGRIHGT_METRICS>>'''
(WORK / "agrihgt_metrics.py").write_text(METRICS_SCRIPT)
best_run = f"{ABLATION_BACKBONE}_L{NUM.get('ablation_depth_selected_by_val', 2)}"
subprocess.run([sys.executable, str(WORK / "agrihgt_metrics.py"), "--pred-dir", str(PRED_DIR),
                "--out-dir", str(WORK / "tables"), "--best", best_run, "--n-classes", str(N_CLASSES)], check=True)
print(pd.DataFrame(RUNS).sort_values("run_id").to_string(index=False))

# %%
# ---------------------------------------------------------------- 14. package everything for download
NUM["runs"] = RUNS
save_numbers()
out_zip = WORK.parent / "agrihgt_results.zip"
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for sub in ["predictions", "tables"]:
        for f in (WORK / sub).glob("*"):
            zf.write(f, f"{sub}/{f.name}")
    for f in ["paper_numbers.json", "runs.csv", "split.csv"]:
        if (WORK / f).exists():
            zf.write(WORK / f, f)
    for f in (WORK / "features").glob("*.json"):
        zf.write(f, f"features/{f.name}")
print("Download:", out_zip)
print(json.dumps({k: v for k, v in NUM.items() if k not in ("runs", "original_per_class", "disease_categories")}, indent=1)[:4000])
