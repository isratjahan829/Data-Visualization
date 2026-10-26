# Methodology Figure — Format & Structure Reference

Reference layout, saved from a sample research methodology figure (retinal disease
classification with MAE + CBAM + GRU). Use it as a template for drawing new
methodology / workflow diagrams.

## 1. Overall Format

- **Orientation:** Portrait, single page (about 1300 × 1600 px, ~4:5).
- **Flow:** Serpentine (snake) layout, top → bottom:
  - Row 1 left → right, then down the right side
  - Row 2 right → left
  - Row 3 left → right (model blocks)
  - Row 4 right → left (evaluation)
  - Row 5: XAI (bottom left) → Web App (right)
- **Arrows:** Thick black arrows with sharp elbow (orthogonal) connectors between blocks.
- **Block style:** Each stage is a rectangle with a light pastel fill and a colored border.
  A **blue rounded "pill" title tab** (white bold text) sits centered on the top edge.
- **Side labels:** The two main model stages use a **tall vertical blue rounded banner**
  with white bold rotated text (left side and right side).
- **Font:** Sans-serif, bold for headings and labels.

## 2. Block-by-Block Structure

| # | Block (title tab) | Position | Fill / border | Contents |
|---|-------------------|----------|---------------|----------|
| 1 | **Sample Data** | Row 1, left | Light pink fill, red border | 3 database cylinder icons (D1, D2, D3, magenta) merging with arrows into a grid of sample images (4 × 4 + 1) with class names under each image |
| 2 | **Image Preprocessing** | Row 1, middle | Light pink fill, red border | 3 chevron/pentagon labels in sequence: *Resizing* (blue) → *Contrast Enhancement* (yellow) → *Noise Reduction* (green); one example image under each step |
| 3 | **Data Augmentation** | Row 1, right | Pink fill, magenta border | Vertical stack of green-gray buttons: Rotation, Flipping, Brightness Adjustment, Contrast Adjustment, Mixup, Cutmix, Noise addition |
| 4 | **Output of augmentation samples** | Row 2, right | Light yellow-green fill, green border | 3 × 3 grid of augmented images, each with a small caption (Original, Rotation ±, Flip, Brightness, Contrast, Mixup, Cutmix …) |
| 5 | **Feature Extraction and Selection** | Row 2, middle | Light orange/cream fill | Inner purple rounded panel: *LBP* (stacked yellow cards) → *Texture Feature*; *Morphological Features* (green) → *Eccentricity* (yellow), *Compactness* (blue), *Circularity* (pink). Below: *Dimension Reduction (PCA)* → *Feature Ranking (Random Forest)* → *Final Feature Set* (teal boxes) |
| 6 | **Masked Autoencoder Model** (vertical left banner) | Row 3, left | White fill, black border | Masked patch grid → patch strip → Encoder (gray block) → blue token column → masked tokens → Decoder (gray block) → red token column → reconstructed patch grid; small "Pretrain" label at top right |
| 7 | **CBAM and Improved CBAM and GRU Architecture** (vertical right banner) | Row 3, right | White fill, black border | Top: Input Feature (blue 3D cube) → Channel Attention Module (green box) → ⊗ → Spatial Attention Module (purple box) → ⊗ → Refined Feature (blue cube). Bottom: patch grid → patch strip → CBAM (orange) → Encoder (peach); right side: Image patches → *Linear Projection of Flattened* → Patches Encoding (Norm → MSA → ⊕ → Norm → MLP → ⊕, green boxes) |
| 8 | **Evaluation** | Row 4, left | Gradient fill (green → purple) | Two columns of cream arrow/chevron boxes: Accuracy / Precision, F1 Score / Recall, PR AUC / AUC-ROC, MCC / Specificity, 10-Fold Cross Validation / Log Loss; one centered: Cohen's Kappa |
| 9 | **XAI Implementation** | Row 5, left | White fill, black border | Two image groups (7 + 6 classes). For each class: class name on top, original image, Grad-CAM heatmap below |
| 10 | **Web Application** | Row 4–5, right | Light green fill, green border | *Best Performing Model* (purple box) → *Flask Framework* (purple box), both feeding down into 2 app screenshots (heatmap + upload panel + class probability bars + Predict / Reload buttons) |

## 3. Connection Order (arrows)

```
Sample Data ──► Image Preprocessing ──► Data Augmentation
                                              │
                                              ▼
Feature Extraction & Selection ◄── Output of Augmentation Samples
        │
        ▼
Masked Autoencoder Model ──► CBAM + Improved CBAM + GRU
                                              │
                                              ▼
                   Evaluation ◄───────────────┘
                        │
                        ▼
                XAI Implementation ──► Web Application
```

## 4. Color Palette (approximate)

| Use | Color |
|-----|-------|
| Title tabs / vertical banners | Bright blue `#1E9BE0`, white text |
| Data / preprocessing panels | Light pink `#F8D7D7`, red border `#E05050` |
| Augmentation panel | Pink `#F4C6D8`, magenta border `#C83CC8` |
| Augmented output panel | Pale yellow-green `#EEF5C8`, green border `#8CC63F` |
| Feature panel | Cream `#FBEFD5`; inner panel lavender `#E3D5F0` |
| Feature chips | Yellow `#FFE45C`, green `#D7F0B0`, blue `#CFE6F7`, pink `#F8D0D8`, teal `#9BD8E0` |
| Evaluation background | Gradient green `#2EE6A8` → purple `#B39DF0`; boxes cream `#FFF3C8` |
| Web app panel | Light green `#D8EBA0`; buttons purple `#7A6491` |
| Attention modules | Channel = green border, Spatial = purple border, feature cubes = blue `#4A90D9` |
| Arrows / borders | Black, thick (≈3 px) |

## 5. Reuse Checklist

1. Keep 5 horizontal rows in a snake flow, with thick black elbow arrows.
2. Give every block a pastel panel plus a blue pill title tab.
3. Use vertical blue banners for the main model blocks.
4. Show real sample images inside the data, preprocessing, augmentation and XAI blocks.
5. Show evaluation metrics as paired chevron boxes on a gradient background.
6. End with deployment (web app screenshots) at the bottom right.
