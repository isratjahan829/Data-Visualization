// Build journal-specific manuscripts: node journal.js <sr|ieee> <out.docx> <figdir>
const { d, table, numbering, styles, runs } = require('./lib');
const { Document, Packer, Paragraph, TextRun, AlignmentType, PageBreak, Footer, PageNumber, ImageRun, HeadingLevel } = d;
const fs = require('fs');
const C = require('./content');
const [STYLE, OUT, FIGDIR] = process.argv.slice(2);
const SR = STYLE === 'sr';

// ---------- Tables registry ----------
const TABLES0 = {
  lit: { title: 'Reported cross-participant results in the literature.', note: 'Values are as reported by the original studies and are not directly comparable: datasets, features, splitting, use of target-participant data and reporting unit (participant mean or best fold) differ. [[Verify every value against the original paper.]]', header: ['Study', 'Dataset(s)', 'Approach', 'Reported accuracy'], rows: C.LIT_TABLE, widths: [2000, 2000, 3226, 1800] },
  protocol: { title: 'Dataset, features and evaluation protocol.', header: ['Item', 'Setting'], rows: C.PROTOCOL_TABLE, widths: [2200, 6826] },
  models: { title: 'Model families.', header: ['Family', 'Models', 'Role'], rows: C.MODEL_TABLE, widths: [2200, 4426, 2400] },
  results: { title: 'Cross-participant four-class emotion recognition on SEED-IV, zero-shot and after 20-window calibration.', note: 'Values are means over 15 held-out participants (each averaged over three seeds) with 95% bootstrap confidence intervals. Chance balanced accuracy is 0.25. Gain = calibrated minus zero-shot balanced accuracy, in percentage points.', header: ['Model', 'Zero-shot balanced accuracy', 'Calibrated balanced accuracy', 'Calibrated macro F1', 'Gain (points)'], rows: C.RESULTS_TABLE, widths: [2226, 1950, 1950, 1450, 1450], group: true },
  paired: { title: 'Paired participant-level comparisons of calibrated CLISA-EWU with other models.', note: 'Differences are CLISA-EWU minus comparator in percentage points of balanced accuracy, with 95% bootstrap confidence intervals over participants. Two-sided Wilcoxon signed-rank tests; Holm adjustment over 14 comparisons.', header: ['Comparison', 'Difference (95% CI)', 'Participants favouring CLISA-EWU', 'Wilcoxon p', 'Holm p'], rows: C.PAIRED_TABLE, widths: [2426, 2300, 1600, 1350, 1350], group: true },
  cluster: { title: 'Clustering of raw features and CLISA-EWU embeddings.', note: 'Higher silhouette and Calinski–Harabasz and lower Davies–Bouldin values indicate tighter clusters. CLISA-EWU embeddings were computed with the best-run model on all 15 participants (14 of whom were in its training set). The thesis draft printed the participant silhouettes of the CLISA-EWU embeddings with the wrong sign; corrected values are shown.', header: ['Feature space', 'Labels (clusters)', 'Silhouette', 'Davies–Bouldin', 'Calinski–Harabasz'], rows: C.CLUSTER_TABLE, widths: [2626, 1900, 1400, 1600, 1500] },
};
const TABLES = { ...TABLES0, ...C.TABLES_EXT };
for (const t of Object.values(TABLES)) { const W = 9026; const rel = t.rel || t.widths; const tot = rel.reduce((a, b) => a + b, 0); t.widths = rel.map((r) => Math.floor((r * W) / tot)); t.widths[t.widths.length - 1] += W - t.widths.reduce((a, b) => a + b, 0); }
const SR_SUPP_FIGS = new Set(['dataset', 'split', 'classical', 'violin', 'conf', 'tsne', 'loss', 'abl', 'neuro']);
const SR_SUPP_TABS = new Set(['lit', 'protocol', 'models', 'cluster']);
const isSuppF = (k) => SR && SR_SUPP_FIGS.has(k);
const isSuppT = (k) => SR && SR_SUPP_TABS.has(k);

// ---------- Document skeleton (blocks) ----------
const B = [];
const h1 = (t) => B.push({ t: 'h1', text: t });
const h2 = (t) => B.push({ t: 'h2', text: t });
const p = (text) => B.push({ t: 'p', text });
const eqb = (key) => B.push({ t: 'eq', key, text: '' });
const mixed = (arr) => arr.forEach(([t, x]) => (t === 'eq' ? eqb(x) : p(x)));
const bullets = (items) => items.forEach((text) => B.push({ t: 'b', text }));

let s1 = 0, s2 = 0;
const H1 = (t) => { if (SR) return h1(t); s1++; s2 = 0; h1(`${s1}  ${t.toUpperCase()}`); };
const H2 = (t) => { if (SR) return h2(t); s2++; h2(`${s1}.${s2}  ${t}`); };

if (SR) {
  H1('Introduction'); C.INTRO.forEach(p);
  p('Here we address these questions with a controlled benchmark of 26 models, report zero-shot and calibrated performance separately at the participant level, and quantify the cost of a six-electrode wearable-compatible montage.');
  H1('Results');
  H2('Benchmark overview'); p(C.R_OVERVIEW);
  H2('Zero-shot cross-participant decoding'); p(C.R_ZS);
  H2('Few-shot calibration changes the ranking of models'); C.R_CAL.forEach(p);
  H2('Six electrodes retain most calibrated performance'); p(C.R_MONTAGE);
  H2('Variability across participants'); p(C.R_VAR);
  H2('Class-level behaviour and electrode importance'); p(C.R_CLASS); p(C.R_SHAP);
  H2('Representation geometry'); p(C.R_REPR);
  H2('Secondary analyses'); p(C.R_SECONDARY);
  H1('Discussion'); C.DISCUSSION.forEach(p); p(C.CONCLUSION);
  H1('Methods');
  H2('Dataset and features'); C.M_DATA.forEach(p);
  H2('Evaluation protocol'); C.M_PROTOCOL.forEach(p);
  H2('Models'); C.M_MODELS.forEach(p);
  H2('CLISA-EWU'); C.M_CLISA.forEach(p);
  H2('Representation, attribution and statistical analysis'); C.M_ANALYSIS.forEach(p);
  H2('Ethics declarations'); p('This study used only the publicly available, de-identified SEED-IV dataset. The original data collection was approved by the providers’ institutional ethics committee and all participants gave informed consent; no new human data were collected. All methods were carried out in accordance with relevant guidelines and regulations.');
} else {
  H1('Introduction'); C.INTRO.forEach(p);
  p('The main contributions of this work are as follows:'); bullets(C.CONTRIB);
  p('Section 2 reviews related work, Section 3 describes materials and methods, Section 4 reports the results, Section 5 discusses their implications and limitations, and Section 6 concludes the paper.');
  H1('Related Work'); C.RELATED.forEach(([t, x]) => { H2(t); p(x); });
  H1('Materials and Methods');
  H2('Dataset and Features'); C.M_DATA.forEach(p);
  H2('Evaluation Protocol'); C.M_PROTOCOL.forEach(p);
  H2('Models'); C.M_MODELS.forEach(p);
  H2('CLISA-EWU'); C.M_CLISA.forEach(p);
  H2('Evaluation Metrics'); mixed(C.METRICS);
  H2('Statistical Analysis'); mixed(C.STATS);
  H2('Representation and Attribution Analysis'); p(C.M_ANALYSIS[0]);
  H1('Results');
  H2('Benchmark Overview'); p(C.R_OVERVIEW);
  H2('Zero-Shot Cross-Participant Decoding'); p(C.R_ZS);
  H2('Effect of Few-Shot Calibration'); C.R_CAL.forEach(p);
  H2('Ranking Across Participants'); p(C.R_RANK);
  H2('Effect Sizes'); p(C.R_EFFECT);
  H2('Electrode Reduction'); p(C.R_MONTAGE);
  H2('Multi-Metric Evaluation'); p(C.R_MULTI);
  H2('Variability Across Participants and Seeds'); p(C.R_VAR); p(C.R_VARDEC);
  H2('Who Benefits From Calibration'); p(C.R_CORR);
  H2('Class-Level Behaviour and Electrode Importance'); p(C.R_CLASS2); p(C.R_SHAP);
  H2('Representation Geometry'); p(C.R_REPR);
  H2('Secondary Analyses'); p(C.R_SECONDARY);
  H1('Discussion'); C.DISCUSSION.slice(0, 3).forEach(p); H2('Limitations'); p(C.DISCUSSION[3]);
  H1('Conclusion'); p(C.CONCLUSION);
}

// ---------- Pass 1: numbering by first mention ----------
const citeOrder = [], figOrder = { main: [], supp: [] }, tabOrder = { main: [], supp: [] };
const TOK = /\{F:(\w+)\}|\{T:(\w+)\}|\{([a-z_,]+)\}/g;
function scan(text) {
  let m; const re = new RegExp(TOK.source, 'g');
  while ((m = re.exec(text))) {
    if (m[1]) { const L = isSuppF(m[1]) ? figOrder.supp : figOrder.main; if (!L.includes(m[1])) { L.push(m[1]); scan(C.FIGS[m[1]].text); } }
    else if (m[2]) { const L = isSuppT(m[2]) ? tabOrder.supp : tabOrder.main; if (!L.includes(m[2])) { L.push(m[2]); scan(TABLES[m[2]].note || ''); m[2] === 'lit' && C.LIT_TABLE.forEach((r) => scan(r[0])); } }
    else m[3].split(',').forEach((k) => { if (!C.REFS[k]) throw new Error('missing ref ' + k); if (!citeOrder.includes(k)) citeOrder.push(k); });
  }
}
B.forEach((b) => scan(b.text));
// Supplementary items cite references too; scan them after the main text so main citations come first
const roman = (n) => ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'][n - 1];
const figLabel = (k) => (figOrder.main.includes(k) ? String(figOrder.main.indexOf(k) + 1) : 'S' + (figOrder.supp.indexOf(k) + 1));
const tabLabel = (k) => (tabOrder.main.includes(k) ? (SR ? String(tabOrder.main.indexOf(k) + 1) : roman(tabOrder.main.indexOf(k) + 1)) : 'S' + (tabOrder.supp.indexOf(k) + 1));

function compress(nums) {
  nums.sort((a, b) => a - b); const out = []; let i = 0;
  while (i < nums.length) { let j = i; while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++; out.push(j - i >= 2 ? `${nums[i]}–${nums[j]}` : j === i ? `${nums[i]}` : `${nums[i]},${nums[j]}`); i = j + 1; }
  return out;
}
function render(text) {
  return text
    .replace(/(Fig\.|Figure) \{F:(\w+)\}/g, (_, w, k) => (isSuppF(k) ? `Supplementary Fig. ${figLabel(k)}` : `${SR ? 'Fig.' : 'Fig.'} ${figLabel(k)}`))
    .replace(/Table \{T:(\w+)\}/g, (_, k) => (isSuppT(k) ? `Supplementary Table ${tabLabel(k)}` : `Table ${tabLabel(k)}`))
    .replace(/\{([a-z_,]+)\}/g, (_, ks) => {
      const nums = ks.split(',').map((k) => citeOrder.indexOf(k) + 1);
      if (SR) return '^' + compress(nums).join(',') + '^';
      return ' ' + compress(nums).map((x) => x.includes('–') ? `[${x.split('–')[0]}]–[${x.split('–')[1]}]` : x.split(',').map((y) => `[${y}]`).join(', ')).join(', ');
    });
}

// ---------- Pass 2: render ----------
const SZ = SR ? 22 : 20;
const para = (text, o = {}) => new Paragraph({ children: runs(render(text), { size: o.size || SZ, ...(o.run || {}) }), spacing: { after: 140, line: SR ? 360 : 300 }, alignment: o.align || AlignmentType.JUSTIFIED, ...(o.p || {}) });
function figBlock(k) {
  const f = C.FIGS[k]; let w = 600, h = Math.round((600 * f.h) / f.w);
  if (h > 640) { h = 640; w = Math.round((640 * f.w) / f.h); }
  const lab = figLabel(k);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160 }, children: [new ImageRun({ type: 'png', data: fs.readFileSync(`${FIGDIR}/${f.file}`), transformation: { width: w, height: h } })] }),
    new Paragraph({ children: [...runs(SR ? `**${isSuppF(k) ? 'Supplementary Figure' : 'Figure'} ${lab}. ${f.title}** ` : `**Fig. ${lab}.** `, { size: 18 }), ...runs(render(SR ? f.text : `${f.title} ${f.text}`), { size: 18 })], spacing: { after: 280, line: 260 }, alignment: AlignmentType.JUSTIFIED }),
  ];
}
function tabBlock(k) {
  const t = TABLES[k]; const lab = tabLabel(k);
  const cap = SR
    ? new Paragraph({ keepNext: true, spacing: { before: 200, after: 100 }, children: runs(`**${isSuppT(k) ? 'Supplementary Table' : 'Table'} ${lab}. ${t.title}**`, { size: 18 }) })
    : new Paragraph({ keepNext: true, alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, children: [new TextRun({ text: `TABLE ${lab}`, size: 18 }), new TextRun({ text: '', break: 1 }), ...runs(render(t.title), { size: 16 })] });
  const rows = t.rows.map((r) => r.map((c) => render(String(c))));
  const out = [cap, table(t.header.map(render), rows, t.widths, { size: t.size || 17, highlightRow: t.group ? (r) => r[1] === '' : undefined })];
  if (t.note) out.push(new Paragraph({ children: runs(render(t.note), { size: 16 }), spacing: { before: 80, after: 240 } }));
  else out.push(new Paragraph({ spacing: { after: 200 }, children: [] }));
  return out;
}

let eqNo = 0; const EQSZ = JSON.parse(fs.readFileSync(`${FIGDIR}/equations/sizes.json`));
const body = [];
const placedF = new Set(), placedT = new Set();
for (const b of B) {
  if (b.t === 'h1') body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(b.text)], spacing: { before: 320, after: 140 } }));
  else if (b.t === 'h2') body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(b.text)], spacing: { before: 220, after: 100 } }));
  else if (b.t === 'b') body.push(new Paragraph({ children: runs(render(b.text), { size: SZ }), numbering: { reference: 'bul', level: 0 }, spacing: { after: 80, line: 300 } }));
  else if (b.t === 'eq') { eqNo++; const [w, h] = EQSZ[b.key]; body.push(new Paragraph({ spacing: { before: 100, after: 100 }, tabStops: [{ type: d.TabStopType.CENTER, position: 4513 }, { type: d.TabStopType.RIGHT, position: 9026 }], children: [new TextRun('\t'), new ImageRun({ type: 'png', data: fs.readFileSync(`${FIGDIR}/equations/eq_${b.key}.png`), transformation: { width: Math.round(w * 0.3), height: Math.round(h * 0.3) } }), new TextRun({ text: `\t(${eqNo})`, size: SZ })] })); }
  else body.push(para(b.text));
  if (b.t === 'p' || b.t === 'b') {
    for (const m of b.text.matchAll(/\{T:(\w+)\}/g)) { const k = m[1]; if (!isSuppT(k) && !placedT.has(k)) { placedT.add(k); body.push(...tabBlock(k)); } }
    for (const m of b.text.matchAll(/\{F:(\w+)\}/g)) { const k = m[1]; if (!isSuppF(k) && !placedF.has(k)) { placedF.add(k); body.push(...figBlock(k)); } }
  }
}

// ---------- Front matter ----------
const front = [];
const draftNote = '**Drafting note (delete before submission).** Yellow-highlighted text marks results that are still pending or facts to verify. Every number comes from the per-fold results in the project repository, recomputed at the participant level. All thesis figures are retained; the original Welch t-test figure was replaced by a corrected paired-test figure.';
if (SR) {
  front.push(
    new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: C.TITLE, bold: true, size: 34 })] }),
    para(C.AUTHORS, { align: AlignmentType.LEFT }), para(C.AFFIL, { align: AlignmentType.LEFT, size: 20 }),
    para(draftNote, { size: 18, align: AlignmentType.LEFT }),
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun('Abstract')], spacing: { before: 240, after: 120 } }),
    para(C.ABSTRACT_SR),
    para('**Keywords:** ' + C.KEYWORDS.join(', '), { align: AlignmentType.LEFT }),
  );
} else {
  front.push(
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [new TextRun({ text: C.TITLE, size: 40 })] }),
    para(C.AUTHORS.replace(/\^1\*?\^/g, '').replace(' & ', ', and '), { align: AlignmentType.CENTER, size: 22 }),
    para(draftNote, { size: 18, align: AlignmentType.LEFT }),
    new Paragraph({ children: [...runs('**Abstract**—', { size: 18 }), ...runs(C.ABSTRACT_IEEE, { size: 18 })], alignment: AlignmentType.JUSTIFIED, spacing: { after: 140, line: 260 } }),
    new Paragraph({ children: [...runs('**Index Terms**—', { size: 18 }), new TextRun({ text: C.KEYWORDS.map((k, i) => (i ? k : k[0].toUpperCase() + k.slice(1))).join(', '), size: 18 })], spacing: { after: 140 } }),
    new Paragraph({ children: runs('S. Khan, O. B. Qashem, K. M. F. A. Bari, R. Rahman and A. W. Reza are with the Department of Computer Science and Engineering, East West University, Dhaka 1212, Bangladesh. Corresponding author: S. Khan (e-mail: saifkhantdm@gmail.com). Manuscript received [[date]].', { size: 16, italics: true }), spacing: { after: 240 } }),
  );
}

// ---------- Back matter ----------
const back = [];
const bh = (t) => back.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(SR ? t : t.toUpperCase())], spacing: { before: 320, after: 140 } }));
const DATA = 'SEED-IV is available from its providers under their licence (https://bcmi.sjtu.edu.cn/home/seed/). Data splits, per-fold predictions, confusion matrices and result tables are available at https://github.com/saifkhancse/clisa-ewu-brain-informatics [[archive a tagged release on Zenodo and cite its DOI]].';
const CODE = 'Model, training, evaluation and analysis code is available at https://github.com/saifkhancse/clisa-ewu-brain-informatics, including the scripts that regenerate every statistic and figure in this paper.';
const CONTRIB = 'S.K. conceived the study, developed the models, ran the experiments and drafted the manuscript. O.B.Q. contributed to experiments, analysis and manuscript preparation. K.M.F.A.B. contributed to implementation, evaluation and documentation. R.R. contributed to experiments, reporting and software integration. A.W.R. supervised the project and critically revised the manuscript. All authors reviewed and approved the final manuscript.';
const ACK = 'The authors thank the providers of the SEED-IV dataset and the Department of Computer Science and Engineering, East West University. No external funding was received [[confirm]].';
const refPara = (k, i) => new Paragraph({ children: runs((SR ? `${i + 1}. ` : `[${i + 1}] `) + C.REFS[k], { size: SR ? 20 : 16 }), indent: { left: 500, hanging: 500 }, spacing: { after: 60, line: 260 } });
if (SR) {
  bh('Data availability'); back.push(para(DATA));
  bh('Code availability'); back.push(para(CODE));
  bh('References');
  // references cited only in supplementary items are appended after main ones
  [...figOrder.supp.map((k) => C.FIGS[k].text), ...tabOrder.supp.map((k) => (TABLES[k].note || '') + (k === 'lit' ? C.LIT_TABLE.map((r) => r[0]).join(' ') : ''))].forEach(scan);
  citeOrder.forEach((k, i) => back.push(refPara(k, i)));
  bh('Acknowledgements'); back.push(para(ACK));
  bh('Author contributions'); back.push(para(CONTRIB));
  bh('Competing interests'); back.push(para('The authors declare no competing interests.'));
  bh('Additional information'); back.push(para('Supplementary Information accompanies this paper (below). Correspondence and requests for materials should be addressed to S.K.'));
  back.push(new Paragraph({ children: [new PageBreak()] }));
  back.push(new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: 'Supplementary Information', bold: true, size: 32 })] }));
  back.push(para(`for "${C.TITLE}"`, { align: AlignmentType.LEFT, run: { italics: true } }));
  tabOrder.supp.forEach((k) => back.push(...tabBlock(k)));
  figOrder.supp.forEach((k) => back.push(...figBlock(k)));
  } else {
  bh('Acknowledgments'); back.push(para(ACK));
  bh('Data and Code Availability'); back.push(para(DATA + ' ' + CODE));
  bh('References');
  citeOrder.forEach((k, i) => back.push(refPara(k, i)));
  back.push(para('**Author biographies** [[TAFFC requires a short biography and photograph for each author at the end of the accepted paper.]]', { align: AlignmentType.LEFT }));
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: SR ? 'Arial' : 'Times New Roman', size: SZ } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: SR ? 28 : 20, bold: true, font: SR ? 'Arial' : 'Times New Roman' }, paragraph: { outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: SR ? 23 : 20, bold: SR, italics: !SR, font: SR ? 'Arial' : 'Times New Roman' }, paragraph: { outlineLevel: 1 } },
    ],
  },
  numbering,
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } }, ...(SR ? { lineNumbers: { countBy: 1, restart: d.LineNumberRestartFormat.CONTINUOUS } } : {}) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children: [...front, ...body, ...back],
  }],
});
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(STYLE, 'figs main', figOrder.main.map(figLabel).length, figOrder.main.join(','), '| supp', figOrder.supp.join(','), '| tabs', tabOrder.main.join(','), '/', tabOrder.supp.join(','), '| refs', citeOrder.length, '| unused refs', Object.keys(C.REFS).filter((k) => !citeOrder.includes(k)).join(','));
});
