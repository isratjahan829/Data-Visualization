const { d, P, H1, H2, Bullet, Num, table, Caption, numbering, styles, runs } = require('./lib');
const { Document, Packer, Paragraph, TextRun, AlignmentType, Footer, PageNumber } = d;
const fs = require('fs');
const C = []; const add = (...x) => C.push(...x);

add(new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: 'CLISA-EWU: pre-submission audit and roadmap to a high-impact journal', bold: true, size: 32 })] }));
add(P('Companion to *CLISA-EWU_manuscript_draft_LancetDH.docx*. Sources reviewed: *Saif_Thesis_Springer.docx*, and the full repository github.com/saifkhancse/clisa-ewu-brain-informatics (README, docs/, results/, notebooks/provenance/05_clisa_ewu_main_experiments.ipynb, notebooks/braininfo_better_clisa_embedding_tsne.ipynb, notebooks/08_final_stats_and_figures_v2.ipynb, 00_eda_seediv.ipynb). All statistics were recomputed from results/aggregates/deep_models_seediv/deep_models_all_folds.csv.'));

add(H1('1. Bottom line'));
add(
  P('**The Lancet itself is not a realistic target.** It publishes clinical studies that change practice, and this work involves no patients, no clinical outcome, and no prospective data. Journals with an impact factor above 20 in this space (The Lancet Digital Health, Nature Machine Intelligence, Nature Biomedical Engineering, Nature Human Behaviour) expect, at a minimum, external validation on independent cohorts, an evaluation protocol with no leakage, and either a clearly new method or a finding that changes how the field works.'),
  P('**The strongest honest story in the current data is not "CLISA-EWU is state of the art".** It is: *in cross-participant EEG emotion decoding, the gains attributed to new architectures appear mainly after brief labelled calibration on the new user, and a six-electrode wearable-compatible montage keeps about 91% of calibrated performance.* The new draft is built around that message, because reviewers cannot knock it down and it makes a contribution about methods that matters to the whole field.'),
  P('**With the current evidence only**, realistic targets are IEEE Transactions on Affective Computing (impact factor about 9–10), IEEE Journal of Biomedical and Health Informatics (about 6–7), Journal of Neural Engineering, or npj Digital Medicine (about 12–15, if a wearable or clinical angle is added). **After the work in section 4**, The Lancet Digital Health or Nature Machine Intelligence becomes a defensible attempt. Impact factors are approximate; check the current Journal Citation Reports.'),
);

add(H1('2. Critical problems found in the thesis draft (must fix)'));
add(P('Any one of these would lead to rejection at a top journal. Each is shown with the evidence in the repository.'));
const issues = [
  ['1', 'Headline advantage depends on calibration, which uses labelled test-participant data', 'notebook 05, proto_b_calibrate(): 20 random test windows, all parameters fine-tuned for 10 epochs. Zero-shot CLISA-EWU = 0·485 vs two-layer perceptron 0·477 (p=0·25).', 'Report zero-shot and calibrated results side by side (done in new draft). Never call calibrated results "subject-independent" without qualification.'],
  ['2', 'Calibration and test windows can come from the same trial', 'Calibration windows are drawn with np.random.choice across all 2505 windows; features are de_LDS (temporally smoothed). Adjacent windows share trial-specific signatures.', 'Re-run with trial-held-out calibration: calibrate on the first k trials of session 1, test on sessions 2–3. Add a calibration-budget curve (5–200 windows).'],
  ['3', 'Docstring says "fine-tune only the final classifier" but code unfreezes everything', 'proto_b_calibrate sets requires_grad=True for all parameters and calls cal_model.train() (BatchNorm statistics adapt to the new participant).', 'Describe it correctly (done). Run ablation: batch-normalisation statistics only / classifier head only / all parameters.'],
  ['4', 'Model is not "contrastive inter-subject alignment"', 'Stage 1 NT-Xent contrasts two augmentations (noise, channel mask) of the same window; participant and stimulus identity are never used. Original CLISA (Shen et al.) aligns different participants watching the same stimulus.', 'Either implement true cross-participant stimulus-aligned positives (recommended; it may also improve zero-shot results) or rename the method.'],
  ['5', 'Ablation section describes DANCE, not CLISA-EWU', '08_final_stats_and_figures_v2.ipynb: A01 = "Full DANCE (Baseline)". Ablations run under a different protocol (accuracy about 0·53). A04, A07, A08 and A13 give identical numbers (flags not applied).', 'Remove the claim "ablations confirm the full CLISA-EWU design". Run a real CLISA-EWU ablation (no pretraining, noise-only, mask-only, temperature, epochs) under full LOSO.'],
  ['6', 'Clustering result is circular', 'Embeddings computed on all 37 575 windows with the best-fold model, which was trained with labels on 14 of the 15 participants. Emotion silhouette increases trivially.', 'Recompute metrics on the held-out participant only, averaged over all folds.'],
  ['7', 'Sign errors in Table 4 of the thesis', 'Repository: CLISA subject silhouette = –0·098 (62 electrodes) and –0·057 (6). Thesis prints +0·0983 and +0·0574.', 'Correct the signs. Negative subject silhouette actually supports reduced participant clustering.'],
  ['8', 'Best fold compared with other studies’ means', 'Thesis table 1 claims 85·35% beats SS-EMERGE, BFE-Net and AttGraph.', 'Delete. Reviewers treat this as cherry-picking. Report the best fold only in the appendix.'],
  ['9', 'Wrong statistical test', 'Welch t-test on summary statistics treats 45 runs as independent; runs share participants. "p=0·0004" is not valid.', 'Paired Wilcoxon test on participant means (n=15) with Holm correction (done; table 2 of draft).'],
  ['10', 'Two different headline means', 'master summary / per-fold file: 0·6781; final_tables/table_5_2: 0·6800 (and SD 0·0901 vs 0·0866).', 'Use one source of truth (per-fold file, 0·678). Remove the stale table or document why it differs.'],
  ['11', 'Validation set is not informative', 'Validation windows come from the same participants and trials as training; median validation F1 = 1·00 for supervised models.', 'Use participant-wise validation (hold out one or two training participants) for early stopping and model selection.'],
  ['12', 'Possible global standardisation leakage', 'dataset_summary.json: features standardised to global mean 0, SD 1 over all 37 575 windows (including the test participant).', 'Fit standardisation on training participants only inside each fold, or per participant (state which).'],
  ['13', 'Transductive models not labelled as such', 'PseudoLabel uses unlabelled held-out EEG (conf_mask on Xte).', 'Mark transductive methods in tables, or exclude them from the zero-shot comparison.'],
  ['15', '"Balanced accuracy" is actually plain accuracy', 'notebook 05, evaluate(): accuracy_score, not balanced_accuracy_score. Recomputed from confusion matrices: CLISA-EWU calibrated 0·678 (accuracy) vs 0·676 (balanced accuracy).', 'Call the metric accuracy (done in the journal drafts); report balanced accuracy, κ and MCC from confusion matrices; store confusion matrices for every model in future runs.'],
  ['14', 'Literature numbers and reference errors', 'Reference [3] cites FACED as an ICASSP paper; FACED is Sci Data 2023;10:740. Thesis literature values (52·82%, 58·87%...) are taken from secondary summaries.', 'Verify every number and reference against the original paper.'],
];
add(table(['#', 'Problem', 'Evidence', 'Fix'], issues, [400, 2300, 3300, 3026], { size: 17 }));

add(H1('3. What the recomputed statistics show'));
add(
  Bullet('Calibrated CLISA-EWU, 62 electrodes: 0·678 (participant-level 95% CI 0·641–0·718); 6 electrodes: 0·620 (0·590–0·648). Retention 91·4% (87·9–95·0). These numbers stand.'),
  Bullet('Calibrated, 62 electrodes: CLISA-EWU beats BYOL (+6·9 points, all 15 participants, Holm p=0·001) and PseudoLabel (+2·1, p=0·016). The difference from DANN (+1·6) is not significant after correction (p=0·062).'),
  Bullet('Calibrated, 6 electrodes: CLISA-EWU is not significantly better than BYOL, PseudoLabel, SimCLR or DANN. The thesis claim of a "strong mean ranking" must be softened.'),
  Bullet('Zero-shot: CLISA-EWU 0·485 vs two-layer perceptron 0·477 (p=0·25). No meaningful architectural advantage without calibration.'),
  Bullet('Calibration gain is +16 to +21 points for CLISA-EWU, DANN, PseudoLabel, BYOL and SimCLR, but only +0·1 to +2·0 points for supervised networks. This is the most interesting and novel observation in the data; explaining it (section 4, item C) could anchor a high-impact paper.'),
);

add(H1('4. Experiments needed for an impact factor 20+ submission'));
add(P('Ordered by priority. Items A–D are essential; E–G turn a solid paper into a high-impact one.'));
const plan = [
  ['A', 'Trial-held-out calibration and calibration-budget curve', 'Removes the leakage objection (problem 2). Curves for 0, 5, 10, 20, 50, 100, 200 windows; calibration from session 1, test on sessions 2–3.', '1–2 weeks GPU'],
  ['B', 'Participant-wise validation and fold-wise standardisation; re-run all deep models', 'Removes problems 11–12. Re-run the 45-run LOSO.', '2–3 weeks'],
  ['C', 'Mechanism ablation of calibration', 'Batch-normalisation statistics only / head only / all parameters, for CLISA-EWU and two-layer perceptron. If batch-normalisation adaptation explains most of the gain, that is a simple, publishable insight (no labels needed).', '1 week'],
  ['D', 'External validation on SEED, SEED-V and FACED (123 participants)', 'Same protocol, plus train on SEED-IV and test on others. FACED gives the sample size that top journals expect.', '3–4 weeks'],
  ['E', 'True cross-participant contrastive alignment', 'Positives = same stimulus and time point across different participants (original CLISA). Test whether zero-shot accuracy improves. This adds a real methods contribution.', '2–3 weeks'],
  ['F', 'Real wearable recording', 'For example 20–40 participants at East West University with a 4–8-channel dry-electrode headband (Muse, OpenBCI, or similar), same film-clip protocol, ethics approval required. This is the biggest step towards Lancet Digital Health.', '2–4 months, including ethics'],
  ['G', 'Health relevance', 'Collaborate with a psychiatry or psychology group: for example a cohort with depression or anxiety scales (PHQ-9, GAD-7), testing whether calibrated affect decoding tracks symptoms. Without some health link, Lancet-family journals will desk-reject.', '6+ months'],
];
add(table(['', 'Experiment', 'Why / design', 'Effort'], plan, [400, 2600, 4626, 1400], { size: 17 }));

add(H1('5. Submission checklist'));
['Follow the TRIPOD+AI checklist (include as supplementary file).', 'Pre-specify primary outcome and analysis; state that the zero-shot vs calibrated comparison was post hoc if it was.', 'Release code as a tagged version with a Zenodo DOI; include the analysis script that regenerates every number in tables 1–2.', 'Move the thesis-specific material (NeuroSync dashboard, SHAP on one fold, classical leaderboard details, best fold) to the appendix or remove it.', 'Rewrite the cover letter around the field-level finding (calibration, not architecture), not around a leaderboard.', 'Prepare a suggested reviewer list from EEG affective computing and clinical machine learning methodology.', 'Ask a native-English academic editor to review the final text.'].forEach((t) => add(Bullet(t)));

add(H1('6. Files in this package'));
[
  'CLISA-EWU_manuscript_draft_LancetDH.docx – full draft in Lancet Digital Health format (structured summary, research-in-context panel, line numbers). Yellow highlights = pending results or facts to verify.',
  'figure1_zeroshot_vs_calibrated.png, figure2_per_participant_montage.png – new figures (300 dpi) generated from per-fold data.',
  'analysis/stats.py and analysis/figs.py – scripts that regenerate every number and figure; run with the per-fold CSV path as the argument.',
  'analysis/summary_table.csv, analysis/paired_tests.csv – recomputed results.',
].forEach((t) => add(Bullet(t)));

const doc = new Document({ styles, numbering, sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1200, bottom: 1200, left: 1200, right: 1200 } } }, footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) }, children: C }] });
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(process.argv[2], b); console.log('ok'); });
