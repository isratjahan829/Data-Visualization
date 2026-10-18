// Scientific Reports manuscript laid out after the published-article style of the sample paper.
// node srstyle.js <out.docx> <figdir>
const d = require('docx');
const { Document, Packer, Paragraph, TextRun, AlignmentType, Footer, PageNumber, ImageRun, Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType, TabStopType, LevelFormat,
  Math: OMath, MathRun, MathFraction, MathSum, MathSubScript } = d;
const fs = require('fs');
const C = require('./content');
const [OUT, FIGDIR] = process.argv.slice(2);

// ---------- Geometry (DXA) ----------
const PAGE_W = 11906, ML = 1300, MR = 860;
const TEXT_W = PAGE_W - ML - MR;          // full width used by figures and tables
const IND = 1700;                         // left indent of the text column
const COL_W = TEXT_W - IND;
const SERIF = 'Times New Roman', SANS = 'Arial';
const BODY = 20;                          // 10 pt

// ---------- Inline markup ----------
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\[\[[^\]]+\]\]|\^[^^]+\^|~[^~]+~)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith('**')) out.push(new TextRun({ text: t.slice(2, -2), ...base, bold: true }));
    else if (t.startsWith('[[')) out.push(new TextRun({ text: '[' + t.slice(2, -2) + ']', ...base, highlight: 'yellow' }));
    else if (t.startsWith('^')) out.push(new TextRun({ text: t.slice(1, -1), ...base, superScript: true, ...(/^[\d,–]+$/.test(t.slice(1, -1)) ? { color: '1F4E9A' } : {}) }));
    else if (t.startsWith('~')) out.push(new TextRun({ text: t.slice(1, -1), ...base, subScript: true }));
    else out.push(new TextRun({ text: t.slice(1, -1), ...base, italics: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

// ---------- Tables registry ----------
const TABLES0 = {
  lit: { title: 'Cross-participant EEG emotion recognition results reported in the literature. Values are as reported and are not directly comparable, because datasets, features, splitting, use of target-participant data and reporting unit differ [[verify every value against the original paper]].', header: ['Study', 'Dataset(s)', 'Approach', 'Reported accuracy'], rows: C.LIT_TABLE, widths: [2300, 2100, 3102, 2100] },
  protocol: { title: 'Dataset, feature and evaluation protocol summary.', header: ['Item', 'Setting'], rows: C.PROTOCOL_TABLE, widths: [2400, 7202] },
  models: { title: 'Model families evaluated in this study.', header: ['Family', 'Models', 'Role'], rows: C.MODEL_TABLE, widths: [2400, 4802, 2400] },
  results: { title: 'Cross-participant four-class emotion recognition on SEED-IV, zero-shot and after 20-window calibration. Values are means over 15 held-out participants (each averaged over three seeds) with 95% bootstrap confidence intervals; chance balanced accuracy is 0.25. Gain = calibrated minus zero-shot balanced accuracy (percentage points).', header: ['Model', 'Zero-shot balanced accuracy', 'Calibrated balanced accuracy', 'Calibrated macro F1', 'Gain (points)'], rows: C.RESULTS_TABLE, widths: [2402, 2100, 2100, 1500, 1500], group: true },
  paired: { title: 'Paired participant-level comparisons of calibrated CLISA-EWU with other models. Differences are CLISA-EWU minus comparator in percentage points of balanced accuracy with 95% bootstrap confidence intervals; two-sided Wilcoxon signed-rank tests with Holm adjustment over 14 comparisons.', header: ['Comparison', 'Difference (95% CI)', 'Participants favouring CLISA-EWU', 'Wilcoxon p', 'Holm p'], rows: C.PAIRED_TABLE, widths: [2602, 2300, 1900, 1400, 1400], group: true },
  cluster: { title: 'Clustering metrics for raw features and CLISA-EWU embeddings. Higher silhouette and Calinski–Harabasz and lower Davies–Bouldin values indicate tighter clusters. Embeddings were computed with the best-run model on all 15 participants (14 of whom were in its training set).', header: ['Feature space', 'Labels (clusters)', 'Silhouette', 'Davies–Bouldin', 'Calinski–Harabasz'], rows: C.CLUSTER_TABLE, widths: [2802, 2000, 1500, 1600, 1700] },
};

const TABLES = { ...TABLES0, ...C.TABLES_EXT };
for (const t of Object.values(TABLES)) { const rel = t.rel || t.widths; const tot = rel.reduce((a, b) => a + b, 0); t.widths = rel.map((r) => Math.floor((r * TEXT_W) / tot)); t.widths[t.widths.length - 1] += TEXT_W - t.widths.reduce((a, b) => a + b, 0); }
// ---------- Extra text specific to this layout ----------
const CONTRIB_SR = [
  ['Controlled cross-participant benchmark', 'Ten classical and 16 deep models are evaluated on SEED-IV under the same repeated leave-one-participant-out protocol, giving 45 runs per model and electrode montage, with all per-fold outputs released.'],
  ['Zero-shot versus calibrated evaluation', 'Every held-out participant is scored without any of their data and after calibration with 20 labelled windows, showing that the advantage of representation-learning objectives appears mainly after calibration.'],
  ['Wearable electrode reduction', 'The same models are evaluated with a 62-electrode montage and a six-electrode frontal–temporal subset (FP1, FP2, F7, F8, T7, T8), and the retained performance is quantified at the participant level.'],
  ['Rigorous statistics and interpretation', 'Participant-level bootstrap confidence intervals and paired Wilcoxon tests with Holm correction are complemented by class-level, representation-geometry and attribution analyses and a software prototype for wearable use.'],
];
const ORG = 'The remainder of this paper is organized as follows. The next section reviews related work. The Methodology section describes the dataset, evaluation protocol, models, evaluation metrics and statistical analysis. The Results and discussion section presents the benchmark, the effect of calibration, electrode reduction, class-level and representation analyses, and the limitations of the study. The final section concludes the paper.';
const METH_INTRO = 'In this section, the dataset, evaluation protocol, model families, the CLISA-EWU architecture, evaluation metrics and statistical analysis are described. The overall workflow of the study is shown in Fig. {F:workflow}.';
const STRAT1 = 'The trained model predicts all 2505 windows of the held-out participant without access to any of their labels or unlabelled EEG (except for PseudoLabel, which uses unlabelled held-out EEG during training by design). This protocol measures true cross-participant generalization.';
const STRAT2 = 'Twenty windows of the held-out participant are drawn at random, all network parameters are fine-tuned on them for 10 epochs (Adam, learning rate 10^−3^), and performance is measured on the remaining 2485 windows. Twenty windows correspond to about 80 s of EEG, or 0.8% of the participant’s data, and emulate a short user-calibration session.';
const CAVEAT = 'Because calibration windows were sampled across trials, a calibration window and a test window could come from the same film-clip trial; with temporally smoothed features such windows are correlated, so calibrated estimates may be optimistic{liblock} [[pending: trial-held-out calibration (calibrate on session 1, test on sessions 2–3) and a calibration-budget curve of 5–200 windows]]. Validation windows came from the same participants and trials as training windows, so validation scores were near ceiling (median macro F1 1.00 for supervised models) and early stopping effectively selected late epochs; this does not compromise the independence of the held-out participant.';
const TRAINCFG = 'Stage 1 uses AdamW (learning rate 10^−3^, weight decay 10^−4^, cosine annealing, 50 epochs, batch size 128). Stage 2 uses AdamW (learning rate 3 × 10^−4^, weight decay 10^−4^, cosine annealing, up to 50 epochs, early stopping with patience 15 on validation macro F1, class-balanced sampling). Supervised baselines were trained for up to 100 epochs with the same early-stopping rule. Mixed-precision training was used throughout.';
const IMPL = 'All experiments were implemented in Python 3 with PyTorch [[version]] (deep models), scikit-learn [[version]] and XGBoost [[version]] (classical models), and SciPy and statsmodels (statistics), and were run on an NVIDIA GeForce RTX 3050 Laptop GPU [[confirm hardware]].';
const METRICS_INTRO = 'Performance is assessed with complementary class-balanced metrics. For class *i* of *C* = 4 classes, TP~i~, FP~i~ and FN~i~ denote true positives, false positives and false negatives. Per-class recall and precision are';
const METRICS_MID = 'Balanced accuracy, the primary metric, is the mean per-class recall and gives equal weight to each emotion regardless of class frequency (chance level 0.25):';
const METRICS_F1 = 'Macro F1, the secondary metric, is the unweighted mean of per-class F1 scores:';
const METRICS_RET = 'Electrode-reduction retention for a model is the ratio of its six-electrode to its 62-electrode balanced accuracy:';
const NTXENT_INTRO = 'For a batch of *N* windows, the two augmented views give 2*N* projections *z*. For a positive pair (*i*, *j*), the NT-Xent loss is';
const NTXENT_AFTER = 'where sim(·,·) is cosine similarity and τ = 0.5 is the temperature; the loss is averaged over all positive pairs in the batch.';
const RD_INTRO = 'In this section, the results of the benchmark are presented. Zero-shot performance is reported first, followed by the effect of few-shot calibration, the paired statistical comparisons, electrode reduction, variability across participants, class-level behaviour, representation geometry and secondary analyses. The implications and limitations of the findings are discussed at the end of the section.';
const LIT_DISC = 'The calibrated mean of CLISA-EWU (67.8%) lies within the range of published cross-participant SEED-IV results (Table {T:lit}), and the zero-shot mean (48.5%) lies below it. Because the published values differ in their use of target-participant data and in reporting unit, they cannot be ranked against the present results; a fair comparison requires re-evaluating published architectures under both protocols used here.';
const ETHICS = 'This study used only the publicly available, de-identified SEED-IV dataset. Data collection was approved by the providers’ institutional ethics committee and all participants gave informed consent; no new human data were collected. All methods were carried out in accordance with relevant guidelines and regulations.';

// ---------- Blocks ----------
const B = [];
const H1 = (t) => B.push({ t: 'h1', text: t });
const H2 = (t) => B.push({ t: 'h2', text: t });
const H3 = (t) => B.push({ t: 'h3', text: t });
const P = (text) => B.push({ t: 'p', text });
const EQ = (key) => B.push({ t: 'eq', key });

C.INTRO.forEach((x) => P(x.replace(' (Table {T:lit})', '')));
P('This work addresses these gaps through a controlled benchmark with a rigorous, participant-level evaluation. The main contributions of this research are:');
CONTRIB_SR.forEach(([a, b]) => B.push({ t: 'num', text: `*${a}*: ${b}` }));
P(ORG);
H1('Related work');
C.RELATED.forEach(([, x]) => P(x));
H1('Methodology'); P(METH_INTRO);
H2('Dataset and features'); C.M_DATA.forEach(P); P(ETHICS);
H2('Evaluation protocol'); P(C.M_PROTOCOL[0]);
H3('Strategy 1: zero-shot evaluation'); P(STRAT1);
H3('Strategy 2: few-shot calibration'); P(STRAT2); P(CAVEAT);
H2('Model families'); C.M_MODELS.forEach(P);
H2('CLISA-EWU architecture'); C.M_CLISA.forEach(P);
P(NTXENT_INTRO); EQ('ntxent'); P(NTXENT_AFTER);
H3('Training configuration'); P(TRAINCFG);
H3('Implementation'); P(IMPL);
H2('Evaluation metrics'); C.METRICS.forEach(([t, x]) => (t === 'eq' ? EQ(x) : P(x)));
H2('Statistical analysis'); C.STATS.forEach(([t, x]) => (t === 'eq' ? EQ(x) : P(x)));
H2('Representation and attribution analysis'); P(C.M_ANALYSIS[0]);
H1('Results and discussion'); P(RD_INTRO);
H2('Benchmark overview'); P(C.R_OVERVIEW.replace('The experimental workflow and the CLISA-EWU architecture are shown in Fig. {F:workflow} and Fig. {F:arch}. ', ''));
H2('Zero-shot cross-participant decoding'); P(C.R_ZS);
H2('Effect of few-shot calibration'); C.R_CAL.forEach(P);
H2('Ranking across participants'); P(C.R_RANK);
H2('Effect sizes'); P(C.R_EFFECT);
H2('Electrode reduction'); P(C.R_MONTAGE);
H2('Multi-metric evaluation'); P(C.R_MULTI);
H2('Variability across participants and seeds'); P(C.R_VAR); P(C.R_VARDEC);
H2('Who benefits from calibration'); P(C.R_CORR);
H2('Class-level behaviour and electrode importance'); P(C.R_CLASS2); P(C.R_SHAP);
H2('Representation geometry'); P(C.R_REPR);
H2('Secondary analyses'); P(C.R_SECONDARY);
H2('Comparison with published studies'); P(LIT_DISC);
H2('Implications'); P(C.DISCUSSION[1]); P(C.DISCUSSION[2]);
H2('Limitations'); P(C.DISCUSSION[3]);
H1('Conclusion'); P(C.DISCUSSION[0] + ' ' + C.CONCLUSION);

// ---------- Numbering by first mention ----------
const citeOrder = [], figOrder = [], tabOrder = [];
function scan(text) {
  const re = /\{F:(\w+)\}|\{T:(\w+)\}|\{([a-z_,]+)\}/g; let m;
  while ((m = re.exec(text))) {
    if (m[1]) { if (!figOrder.includes(m[1])) { figOrder.push(m[1]); scan(C.FIGS[m[1]].text); } }
    else if (m[2]) { if (!tabOrder.includes(m[2])) { tabOrder.push(m[2]); scan(TABLES[m[2]].title); m[2] === 'lit' && C.LIT_TABLE.forEach((r) => scan(r[0])); } }
    else m[3].split(',').forEach((k) => { if (!C.REFS[k]) throw new Error('missing ref ' + k); if (!citeOrder.includes(k)) citeOrder.push(k); });
  }
}
B.forEach((b) => b.text && scan(b.text));
const compress = (nums) => { nums.sort((a, b) => a - b); const out = []; let i = 0; while (i < nums.length) { let j = i; while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++; out.push(j - i >= 2 ? `${nums[i]}–${nums[j]}` : j === i ? `${nums[i]}` : `${nums[i]},${nums[j]}`); i = j + 1; } return out.join(','); };
const render = (t) => t
  .replace(/(Fig\.|Figure) \{F:(\w+)\}/g, (_, w, k) => `Fig. ${figOrder.indexOf(k) + 1}`)
  .replace(/Table \{T:(\w+)\}/g, (_, k) => `Table ${tabOrder.indexOf(k) + 1}`)
  .replace(/\{([a-z_,]+)\}/g, (_, ks) => '^' + compress(ks.split(',').map((k) => citeOrder.indexOf(k) + 1)) + '^');

// ---------- Paragraph builders ----------
let afterHeading = true;
const body = (text) => {
  const first = afterHeading; afterHeading = false;
  return new Paragraph({ children: runs(render(text), { font: SERIF, size: BODY }), alignment: AlignmentType.JUSTIFIED, indent: { left: IND, firstLine: first ? 0 : 240 }, spacing: { after: 0, line: 252 } });
};
const head = (text, lvl) => {
  afterHeading = true;
  if (lvl === 3) return new Paragraph({ keepNext: true, indent: { left: IND }, spacing: { before: 200, after: 0 }, children: [new TextRun({ text, italics: true, font: SERIF, size: BODY })] });
  return new Paragraph({ keepNext: true, indent: { left: IND }, spacing: { before: lvl === 1 ? 300 : 220, after: 40 }, children: [new TextRun({ text, bold: true, color: '000000', font: SANS, size: lvl === 1 ? 26 : 21 })], heading: lvl === 1 ? d.HeadingLevel.HEADING_1 : d.HeadingLevel.HEADING_2 });
};
const dotted = { bottom: { style: BorderStyle.DOTTED, size: 6, color: 'A6A6A6', space: 6 } };
const caption = (label, text) => new Paragraph({ indent: { left: IND }, border: dotted, spacing: { before: 120, after: 360, line: 240 }, alignment: AlignmentType.LEFT, children: [new TextRun({ text: label, bold: true, font: SERIF, size: 19 }), ...runs('. ' + render(text), { font: SERIF, size: 19 })] });

function figBlock(k) {
  const f = C.FIGS[k]; const maxW = 640, maxH = 560;
  let w = maxW, h = Math.round((maxW * f.h) / f.w);
  if (h > maxH) { h = maxH; w = Math.round((maxH * f.w) / f.h); }
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 240 }, children: [new ImageRun({ type: 'png', data: fs.readFileSync(`${FIGDIR}/${f.file}`), transformation: { width: w, height: h } })] }),
    caption(`Fig. ${figOrder.indexOf(k) + 1}`, `${f.title} ${f.text}`),
  ];
}
const cb = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
const cellBorders = { top: cb, bottom: cb, left: cb, right: cb };
function tabBlock(k) {
  const t = TABLES[k];
  const mk = (txt, w, hdr, grp) => new TableCell({ borders: cellBorders, width: { size: w, type: WidthType.DXA }, shading: hdr ? { fill: 'BFBFBF', type: ShadingType.CLEAR, color: 'auto' } : grp ? { fill: 'E7E6E6', type: ShadingType.CLEAR, color: 'auto' } : undefined, margins: { top: 25, bottom: 25, left: 50, right: 50 }, children: [new Paragraph({ children: runs(render(String(txt)), { font: SERIF, size: t.size || 17, bold: hdr || grp || undefined }) })] });
  const rows = [new TableRow({ tableHeader: true, children: t.header.map((h, i) => mk(h, t.widths[i], true)) }),
    ...t.rows.map((r) => { const grp = t.group && r[1] === ''; return new TableRow({ cantSplit: true, children: r.map((c, i) => mk(c, t.widths[i], false, grp)) }); })];
  return [new Paragraph({ spacing: { before: 200 }, children: [] }), new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: t.widths, rows }), caption(`Table ${tabOrder.indexOf(k) + 1}`, t.title)];
}

// ---------- Equations ----------
const mr = (s) => new MathRun(s);
const sub = (a, b) => new MathSubScript({ children: [mr(a)], subScript: [mr(b)] });
const frac = (n, dd) => new MathFraction({ numerator: n, denominator: dd });
const sumC = (children) => new MathSum({ children, subScript: [mr('i=1')], superScript: [mr('C')] });
const EQS = {
  recall: [sub('Recall', 'i'), mr(' = '), frac([sub('TP', 'i')], [sub('TP', 'i'), mr(' + '), sub('FN', 'i')]), mr(',   '), sub('Precision', 'i'), mr(' = '), frac([sub('TP', 'i')], [sub('TP', 'i'), mr(' + '), sub('FP', 'i')])],
  bacc: [mr('Balanced accuracy = '), frac([mr('1')], [mr('C')]), sumC([sub('Recall', 'i')])],
  f1: [sub('F1', 'i'), mr(' = 2 · '), frac([sub('Precision', 'i'), mr(' · '), sub('Recall', 'i')], [sub('Precision', 'i'), mr(' + '), sub('Recall', 'i')])],
  macrof1: [sub('F1', 'macro'), mr(' = '), frac([mr('1')], [mr('C')]), sumC([sub('F1', 'i')])],
  retention: [mr('Retention = '), frac([sub('BA', '6')], [sub('BA', '62')])],
  ntxent: [sub('ℓ', 'i,j'), mr(' = −log '), frac([mr('exp(sim('), sub('z', 'i'), mr(', '), sub('z', 'j'), mr(')/τ)')], [new MathSum({ children: [mr('exp(sim('), sub('z', 'i'), mr(', '), sub('z', 'k'), mr(')/τ)')], subScript: [mr('k≠i')], superScript: [mr('2N')] })])],
};
let eqNo = 0;
const EQSZ = JSON.parse(fs.readFileSync(`${FIGDIR}/equations/sizes.json`));
const eqPara = (key) => { eqNo++; const [w, h] = EQSZ[key]; const sc = 0.30; return new Paragraph({ indent: { left: IND }, keepLines: true, spacing: { before: 120, after: 120 }, tabStops: [{ type: TabStopType.CENTER, position: Math.round(COL_W / 2) }, { type: TabStopType.RIGHT, position: COL_W }], children: [new TextRun({ text: '\t', font: SERIF }), new ImageRun({ type: 'png', data: fs.readFileSync(`${FIGDIR}/equations/eq_${key}.png`), transformation: { width: Math.round(w * sc), height: Math.round(h * sc) } }), new TextRun({ text: `\t(${eqNo})`, font: SERIF, size: BODY })] }); };

// ---------- Assemble ----------
const out = [];
out.push(
  new Paragraph({ indent: { left: IND }, spacing: { before: 200, after: 240, line: 276, lineRule: d.LineRuleType.AUTO }, children: [new TextRun({ text: C.TITLE, bold: true, font: SANS, size: 46 })] }),
  new Paragraph({ indent: { left: IND }, spacing: { after: 200 }, children: runs(C.AUTHORS.replace(/\^1\*\^/, '^✉^').replace(/\^1\^/g, ''), { font: SANS, size: 21, bold: true }) }),
  new Paragraph({ indent: { left: IND }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 200, line: 252 }, children: runs(C.ABSTRACT_SR, { font: SANS, size: 19, bold: true }) }),
  new Paragraph({ indent: { left: IND }, spacing: { after: 280 }, children: [new TextRun({ text: 'Keywords  ', bold: true, font: SANS, size: 24 }), new TextRun({ text: C.KEYWORDS.map((k, i) => (i ? k : k[0].toUpperCase() + k.slice(1))).join(', '), font: SANS, size: 19 })] }),
  new Paragraph({ indent: { left: IND }, spacing: { after: 200 }, children: runs('[[Drafting note – delete before submission: yellow text marks pending results or facts to verify. All values are recomputed at the participant level from the per-fold results in the project repository.]]', { font: SANS, size: 16 }) }),
);
afterHeading = true;
const placedF = new Set(), placedT = new Set();
for (const b of B) {
  if (b.t === 'h1') out.push(head(b.text, 1));
  else if (b.t === 'h2') out.push(head(b.text, 2));
  else if (b.t === 'h3') out.push(head(b.text, 3));
  else if (b.t === 'eq') out.push(eqPara(b.key));
  else if (b.t === 'num') out.push(new Paragraph({ numbering: { reference: 'contrib', level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { before: 60, after: 60, line: 252 }, children: runs(render(b.text), { font: SERIF, size: BODY }) }));
  else out.push(body(b.text));
  if (b.text) {
    for (const m of b.text.matchAll(/\{T:(\w+)\}/g)) if (!placedT.has(m[1])) { placedT.add(m[1]); out.push(...tabBlock(m[1])); }
    for (const m of b.text.matchAll(/\{F:(\w+)\}/g)) if (!placedF.has(m[1])) { placedF.add(m[1]); out.push(...figBlock(m[1])); }
  }
}
// Back matter
const back = (t, text) => { out.push(head(t, 1)); text && out.push(body(text)); };
back('Data availability', 'The SEED-IV dataset is available from its providers under their licence at https://bcmi.sjtu.edu.cn/home/seed/. Data splits, per-fold predictions, confusion matrices and result tables are available at https://github.com/saifkhancse/clisa-ewu-brain-informatics [[archive a tagged release on Zenodo and cite its DOI]].');
back('Code availability', 'Model, training, evaluation and analysis code, including the scripts that regenerate every statistic and figure in this paper, is available at https://github.com/saifkhancse/clisa-ewu-brain-informatics.');
out.push(head('References', 1));
citeOrder.forEach((k, i) => out.push(new Paragraph({ numbering: { reference: 'refs', level: 0 }, spacing: { after: 20, line: 230 }, alignment: AlignmentType.LEFT, children: runs(C.REFS[k], { font: SERIF, size: 17 }) })));
back('Acknowledgements', 'The authors thank the providers of the SEED-IV dataset and the Department of Computer Science and Engineering, East West University, for their support. All figures were created by the authors; schematic illustrations (Figs. ' + (figOrder.indexOf('workflow') + 1) + ' and ' + (figOrder.indexOf('arch') + 1) + ') were drawn manually and all plots were generated with Python.');
back('Author contributions', 'S.K.: conceptualization, methodology, software, experiments, formal analysis, visualization and writing of the original draft. O.B.Q.: experiments, analysis and manuscript preparation. K.M.F.A.B.: implementation, evaluation and documentation. R.R.: experiments, reporting and software integration (NeuroSync). A.W.R.: supervision, review and editing. All authors reviewed and approved the final manuscript.');
back('Funding', 'No external funding was received for this study [[confirm]].');
back('Declarations');
out.push(head('Competing interests', 2)); out.push(body('The authors declare no competing interests.'));
out.push(head('Additional information', 2));
out.push(new Paragraph({ indent: { left: IND }, spacing: { after: 0, line: 252 }, children: [new TextRun({ text: 'Correspondence', bold: true, font: SERIF, size: BODY }), new TextRun({ text: ' and requests for materials should be addressed to S.K.', font: SERIF, size: BODY })] }));

const affilFooter = new Footer({ children: [
  new Paragraph({ indent: { left: IND }, border: { top: { style: BorderStyle.SINGLE, size: 4, color: '000000', space: 4 } }, spacing: { after: 0 }, children: [new TextRun({ text: 'Department of Computer Science and Engineering, East West University, Dhaka 1212, Bangladesh. ', font: SERIF, size: 17 })] }),
  new Paragraph({ indent: { left: IND }, spacing: { after: 120 }, children: [new TextRun({ text: '✉email: saifkhantdm@gmail.com', font: SERIF, size: 17 })] }),
  new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ children: [PageNumber.CURRENT], font: SANS, size: 16, color: '808080' })] }),
] });
const plainFooter = new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, border: { top: { style: BorderStyle.SINGLE, size: 4, color: '000000', space: 4 } }, children: [new TextRun({ children: [PageNumber.CURRENT], font: SANS, size: 16, color: '808080' })] })] });

const doc = new Document({
  styles: { default: { document: { run: { font: SERIF, size: BODY } } } },
  numbering: { config: [
    { reference: 'contrib', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: IND + 500, hanging: 300 } } } }] },
    { reference: 'refs', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.RIGHT, style: { paragraph: { indent: { left: IND + 360, hanging: 360 } }, run: { font: SERIF, size: 17 } } }] },
  ] },
  sections: [{
    properties: { titlePage: true, page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 1100, bottom: 1100, left: ML, right: MR, footer: 500 } } },
    footers: { first: affilFooter, default: plainFooter },
    children: out,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log('figs', figOrder.length, figOrder.join(','), '| tables', tabOrder.join(','), '| refs', citeOrder.length, '| eqs', eqNo, '| unused', Object.keys(C.REFS).filter((k) => !citeOrder.includes(k)).join(',')); });
