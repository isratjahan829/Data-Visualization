const { d, P, H1, H2, Bullet, table, Caption, Img, numbering, styles, runs } = require('./lib');
const { Document, Packer, Paragraph, TextRun, AlignmentType, PageBreak, Footer, PageNumber, Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle } = d;
const fs = require('fs');
const OUT = process.argv[2];
const FIG = process.argv[3];

const C = [];
const add = (...x) => C.push(...x);
const br = () => new Paragraph({ children: [new PageBreak()] });

// ---------- Title page ----------
add(
  new Paragraph({ children: runs('Article type: Article  |  Target: The Lancet Digital Health (alternative: Nature Machine Intelligence / npj Digital Medicine)', { size: 18, italics: true }), spacing: { after: 240 } }),
  new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 240 }, children: [new TextRun({ text: 'Few-shot calibrated cross-participant decoding of emotional states from full and six-electrode wearable-compatible EEG: a repeated leave-one-participant-out benchmark', bold: true, size: 32 })] }),
  P('Saif Khan, Osama Bin Qashem, K M Fahim A Bari, Rafid Rahman, Ahmed Wasif Reza'),
  P('Department of Computer Science and Engineering, East West University, Dhaka 1212, Bangladesh (S Khan BSc, O B Qashem BSc, K M F A Bari BSc, R Rahman BSc, Prof A W Reza PhD) [[confirm degrees and academic titles]]'),
  P('Correspondence to: Saif Khan, Department of Computer Science and Engineering, East West University, Dhaka 1212, Bangladesh; saifkhantdm@gmail.com'),
  P('Word count (main text): [[~4300; update after final edit]]  |  Tables: 3  |  Figures: 3  |  References: 25'),
  P('**Drafting note (delete before submission).** Text highlighted in yellow marks results that do not yet exist or facts that must be checked. A journal with an impact factor above 20 will not accept this manuscript on the present evidence alone; the accompanying *Pre-submission audit and roadmap* lists the experiments (external validation, trial-held-out calibration, real wearable recording) needed to fill every highlighted slot. No number in this draft has been invented: every value comes from the per-fold result files in the project repository and was recomputed at the participant level.', { run: { size: 20 } }),
  br(),
);

// ---------- Summary ----------
add(H1('Summary'));
add(
  P('**Background** Emotion-aware digital health tools need electroencephalography (EEG) decoders that work for people whose data were never used in training, on headsets with few electrodes. Published cross-participant results are hard to interpret because evaluation protocols, use of target-participant data, and reporting units vary widely. We aimed to quantify how much cross-participant performance depends on few-shot calibration, representation learning, and electrode count under one controlled protocol.'),
  P('**Methods** We analysed the SEED-IV dataset (15 adults, three sessions, 72 film-clip trials per participant; neutral, sad, fear, happy) using 37 575 four-second differential-entropy windows from a 62-electrode montage and a six-electrode frontal–temporal subset (FP1, FP2, F7, F8, T7, T8). Ten classical and 16 deep learning models, including a two-stage contrastive pretraining and fine-tuning model (CLISA-EWU), were evaluated with leave-one-participant-out cross-validation repeated over three seeds (45 runs per model and montage). Each held-out participant was scored zero-shot and after calibration with 20 labelled windows (about 80 s of EEG). The primary outcome was balanced accuracy, summarised per participant (n=15) with bootstrap 95% CIs; models were compared with paired Wilcoxon signed-rank tests with Holm correction. [[External validation on SEED, SEED-V and FACED, and a trial-held-out calibration protocol, pending.]]'),
  P('**Findings** Zero-shot balanced accuracy was modest for every model (best 62-electrode result 0·485, 95% CI 0·428–0·542, CLISA-EWU; chance 0·25) and CLISA-EWU did not differ from a two-layer perceptron (difference 0·9 percentage points, –0·5 to 2·1; p=0·25). After 20-window calibration, CLISA-EWU reached 0·678 (0·641–0·718) with 62 electrodes and 0·620 (0·590–0·648) with six, retaining 91·4% (87·9–95·0) of full-montage performance. Calibration added 16–21 points to contrastive, domain-adversarial, and pseudo-label models but at most 2·0 points to purely supervised networks. With 62 electrodes, CLISA-EWU outperformed BYOL (6·9 points, 5·2–8·5; Holm-adjusted p=0·001) and PseudoLabel (2·1 points, 1·0–3·1; p=0·016) but not DANN after correction; with six electrodes, it did not differ from the next three models. Participant-level calibrated accuracy ranged from 0·541 to 0·813.'),
  P('**Interpretation** In this benchmark, most of the apparent advantage of representation learning for cross-participant EEG emotion decoding appeared only after brief labelled calibration on the new user, and a six-electrode wearable-compatible montage kept about nine-tenths of the calibrated performance. Studies in this field should report zero-shot and calibrated results separately, at the participant level. [[Add one sentence on external validation once available.]] These offline laboratory results do not support clinical use.'),
  P('**Funding** None. [[confirm]]'),
  br(),
);

// ---------- Research in context ----------
const panelCell = (children) => new TableCell({ width: { size: 9026, type: WidthType.DXA }, shading: { fill: 'F2F2F2', type: ShadingType.CLEAR, color: 'auto' }, margins: { top: 160, bottom: 160, left: 200, right: 200 }, borders: { top: { style: BorderStyle.SINGLE, size: 6, color: '7F7F7F' }, bottom: { style: BorderStyle.SINGLE, size: 6, color: '7F7F7F' }, left: { style: BorderStyle.SINGLE, size: 6, color: '7F7F7F' }, right: { style: BorderStyle.SINGLE, size: 6, color: '7F7F7F' } }, children });
const PP = (t) => new Paragraph({ children: runs(t, { size: 20 }), spacing: { after: 120, line: 300 } });
add(new Table({ width: { size: 9026, type: WidthType.DXA }, columnWidths: [9026], rows: [new TableRow({ children: [panelCell([
  PP('**Research in context**'),
  PP('**Evidence before this study**'),
  PP('We searched PubMed, IEEE Xplore, and Google Scholar from Jan 1, 2015, to [[date]] for studies of cross-participant (subject-independent) EEG emotion recognition, using the terms "EEG", "emotion recognition", "cross-subject" OR "subject-independent", "domain adaptation", "contrastive", and "few electrodes" OR "wearable", without language restrictions. [[Run and document the search; give counts screened and included.]] Reported four-class SEED-IV accuracies ranged from about 59% to above 80%, but studies differed in whether target-participant EEG was used (unlabelled for domain adaptation, or labelled for calibration), in data splitting (by window, trial, session, or participant), and in whether results were averaged over participants or reported for a best fold. We found no study that, under one protocol, separated zero-shot from calibrated performance across classical, supervised, self-supervised, adversarial, and semi-supervised models, and quantified the cost of reducing a research montage to a wearable-compatible electrode subset with participant-level uncertainty.'),
  PP('**Added value of this study**'),
  PP('In a benchmark of 26 models with 45 leave-one-participant-out runs each, zero-shot four-class performance was modest and no deep model clearly beat a two-layer perceptron. Calibration with 20 labelled windows (about 80 s) raised balanced accuracy by 16–21 percentage points, but only for models trained with contrastive, adversarial, or pseudo-label objectives, not for purely supervised networks. A six-electrode frontal–temporal montage retained 91% of the calibrated full-montage performance. All results are reported per participant with CIs and paired tests, and all per-fold outputs are openly available.'),
  PP('**Implications of all the available evidence**'),
  PP('Claims that a new architecture "solves" cross-participant EEG emotion recognition should be read alongside whether, and how much, labelled target-participant data were used. Brief user calibration appears to be the main lever for practical systems, and low-density headsets look viable for calibrated use. Before any clinical or consumer deployment, these findings need external validation in independent and more diverse cohorts, on real wearable recordings, and with calibration protocols that keep calibration and test trials separate.'),
])] })] }));
add(br());

// ---------- Introduction ----------
add(H1('Introduction'));
add(
  P('Affective states shape attention, decision making, and mental health, and objective markers of emotion could support digital mental health monitoring, neurofeedback, and adaptive human–computer interfaces.^1,2^ Electroencephalography (EEG) is portable, inexpensive, and has millisecond resolution, which makes it the most plausible neural signal for everyday affect sensing.^2,3^ Two obstacles stand between laboratory results and use. First, a deployed decoder must work for a person whose data were never used in training, but EEG responses to the same stimulus differ greatly between individuals because of anatomy, electrode contact, baseline rhythms, and appraisal.^4–6^ Second, research datasets are recorded with 32–64 electrodes, whereas wearable headsets carry far fewer.'),
  P('Many methods have targeted the first obstacle, including domain-adversarial networks,^5,7^ hemispheric-asymmetry models,^8,9^ graph neural networks,^10,11^ and contrastive learning designed to make representations invariant to the participant.^4^ Reported cross-participant accuracies on SEED-IV range from about 59% to above 80%.^7–11^ These numbers are rarely comparable. Studies differ in whether unlabelled EEG from the test participant is used during training (transductive domain adaptation), whether a few labelled test-participant samples are used to adapt the model (calibration), how windows are split, and whether the reported figure is a mean over participants or the best single fold. Splitting decisions matter because EEG features from adjacent windows of the same trial are strongly correlated, so any overlap between data used for fitting and testing can inflate accuracy.^23–25^'),
  P('We therefore asked three questions under one controlled, fully reported protocol. How well do classical, supervised, self-supervised, adversarial, and semi-supervised models decode four emotions in unseen participants without any target data? How much does a brief labelled calibration change this picture, and does the effect depend on the training objective? How much performance is lost when a 62-electrode research montage is reduced to six frontal–temporal electrodes compatible with a headband-style wearable? We report participant-level estimates with uncertainty and paired statistical comparisons, following the TRIPOD+AI reporting guideline where applicable.^17^'),
);

// ---------- Methods ----------
add(H1('Methods'));
add(H2('Study design and data source'));
add(
  P('This was a secondary analysis of the publicly available SEED-IV dataset.^2^ Fifteen healthy adults [[report sex and age range from the dataset paper]] watched 72 film clips (24 per session, three sessions on different days) selected to elicit neutral, sad, fear, and happy states while 62-channel EEG was recorded with the international 10–20 system. The original study obtained ethics approval and informed consent; no new data were collected for this analysis, and access followed the dataset licence. [[External validation cohorts: SEED (15 participants, three classes),^3^ SEED-V (16 participants, five classes),^22^ and FACED (123 participants, nine classes)^18^ – pending.]]'),
);
add(H2('Features and electrode montages'));
add(
  P('We used the differential-entropy features released with SEED-IV, computed in non-overlapping 4-s windows in five frequency bands (delta, theta, alpha, beta, gamma) and smoothed with a linear dynamical system (de_LDS).^2,3^ This yielded 37 575 windows (2505 per participant; neutral 10 170, sad 10 245, fear 9225, happy 7935). The full montage gave 310 features per window (62 electrodes × five bands). The wearable-compatible montage used six electrodes reachable by a forehead-and-temple band (FP1, FP2, F7, F8, T7, T8), giving 30 features; it was an exact subset of the full feature matrix, so montage comparisons were not confounded by preprocessing. Features were z-standardised before modelling. [[Confirm whether standardisation statistics were computed within each training fold; if they were computed on all 15 participants, re-run with fold-wise statistics.]]'),
);
add(H2('Models'));
add(
  P('We evaluated ten classical classifiers (linear discriminant analysis, support vector machine with radial basis kernel, random forest, extra trees, gradient boosting, XGBoost, k-nearest neighbours, logistic regression, naive Bayes, and a scikit-learn multilayer perceptron) and 16 deep models in five families (appendix table S1): supervised networks (two-layer and deep perceptrons, LSTM, GRU, one-dimensional convolutional network, transformer, EEG Conformer,^16^ and a channel-dropout transformer); a domain-adversarial network (DANN) whose discriminator predicted training-participant identity through a gradient-reversal layer;^5^ self-supervised models (SimCLR^12^ and BYOL^13^); semi-supervised models (PseudoLabel,^14^ which used confident predictions on the unlabelled held-out participant, and MixMatch^15^); and a teacher–student distillation pair (DANCE).'),
  P('CLISA-EWU, the model of primary interest, has an encoder of two fully connected layers (256 and 128 units, batch normalisation, ReLU, dropout 0·3), a two-layer projection head (64 units), and a linear classifier. In stage 1 (50 epochs), the encoder and projector were trained with the normalised temperature-scaled cross-entropy (NT-Xent) loss (temperature 0·5) on two views of each window: one with additive Gaussian noise (SD 0·1) and one with 30% of electrodes masked across all bands. In stage 2 (up to 50 epochs), the whole network was fine-tuned with cross-entropy (AdamW, learning rate 3 × 10^–4^, cosine schedule, class-balanced sampling, early stopping with patience 15 on validation macro F1). Unlike the original CLISA,^4^ which aligns time-locked responses of different participants to the same stimulus, the stage-1 objective here contrasts augmented views of the same window and does not use participant or stimulus identity. [[Either implement the cross-participant stimulus-aligned positive pairs of the original CLISA, or rename the model (eg, "two-stage contrastive pretraining") to avoid overstating subject alignment.]] All models and hyperparameters are listed in appendix table S1, with code in the project repository.'),
);
add(H2('Evaluation protocol'));
add(
  P('We used leave-one-participant-out cross-validation: in each of 15 folds, one participant was held out entirely, and the remaining 14 were pooled and split at random into training (85%) and validation (15%) windows. The procedure was repeated with seeds 1, 7, and 21, giving 45 runs per model and montage. Each held-out participant was scored under two protocols. **Zero-shot**: the trained model predicted all 2505 windows of the held-out participant without access to any of their labels. **Few-shot calibration**: 20 windows of the held-out participant were drawn at random, all network parameters were fine-tuned on them for ten epochs (Adam, learning rate 10^–3^), and performance was measured on the remaining 2485 windows. Twenty windows correspond to about 80 s of EEG, or 0·8% of the participant’s data.'),
  P('Because windows were drawn at random across trials, a calibration window and a test window could come from the same 2–4-min film-clip trial; with LDS-smoothed features, such windows are strongly correlated. The calibrated estimates may therefore be optimistic. [[Pending: trial-held-out calibration – calibrate on the first k trials of session 1 (k = 1, 2, 4 per class) and test only on sessions 2 and 3 – and a calibration-budget curve (5–200 windows).]] In addition, validation windows came from the same participants and trials as training windows, so validation scores were near ceiling (median macro F1 1·00 for supervised models) and early stopping effectively selected late epochs; this does not affect test independence.'),
);
add(H2('Outcomes'));
add(P('The primary outcome was balanced accuracy (mean per-class recall) on the held-out participant; the secondary outcome was macro-averaged F1. Chance balanced accuracy is 0·25. The retention ratio was the six-electrode result divided by the 62-electrode result for the same model.'));
add(H2('Statistical analysis'));
add(
  P('Runs from the same participant are not independent, so we first averaged the three seeds for each participant, giving 15 paired observations per model and montage. We report participant-level means with 95% CIs from 10 000 bootstrap resamples of participants. CLISA-EWU was compared with DANN, PseudoLabel, BYOL, SimCLR, the two-layer and deep perceptrons, and EEG Conformer, separately for each montage and protocol, using two-sided Wilcoxon signed-rank tests; p values were adjusted with the Holm method within each protocol (14 comparisons). The 62-electrode versus six-electrode difference was tested in the same way, and the CI for the retention ratio was obtained by resampling participants. Analyses used Python 3 with SciPy and statsmodels; the analysis script is provided in the repository. [[Optional: a linear mixed model with participant and seed random effects as a sensitivity analysis.]]'),
);
add(H2('Role of the funding source'));
add(P('There was no funding source for this study. [[confirm]]'));

// ---------- Results ----------
add(H1('Results'));
add(
  P('All 26 models completed 45 runs per montage. Table 1 summarises the main deep models; the full leaderboard, including classical classifiers, is in appendix tables S2–S3. The best classical classifier reached a balanced accuracy of 0·484 (XGBoost) with 62 electrodes and 0·441 (linear discriminant analysis) with six electrodes, confirming that differential-entropy features carry emotion information in unseen participants.'),
);
const T1 = [
  ['62 electrodes', '', '', '', ''],
  ['CLISA-EWU', '0·485 (0·428–0·542)', '0·678 (0·641–0·718)', '0·673', '+19·3'],
  ['DANN', '0·458 (0·403–0·514)', '0·662 (0·624–0·703)', '0·654', '+20·4'],
  ['PseudoLabel', '0·468 (0·412–0·524)', '0·657 (0·624–0·693)', '0·654', '+18·9'],
  ['BYOL', '0·431 (0·385–0·479)', '0·609 (0·573–0·645)', '0·594', '+17·8'],
  ['SimCLR', '0·410 (0·367–0·458)', '0·575 (0·537–0·613)', '0·557', '+16·4'],
  ['Two-layer perceptron', '0·477 (0·427–0·527)', '0·496 (0·450–0·544)', '0·493', '+2·0'],
  ['Deep perceptron', '0·479 (0·426–0·533)', '0·492 (0·444–0·542)', '0·489', '+1·3'],
  ['EEG Conformer', '0·453 (0·396–0·511)', '0·459 (0·405–0·513)', '0·454', '+0·7'],
  ['Six electrodes', '', '', '', ''],
  ['CLISA-EWU', '0·406 (0·370–0·439)', '0·620 (0·590–0·648)', '0·611', '+21·4'],
  ['BYOL', '0·440 (0·393–0·491)', '0·612 (0·569–0·653)', '0·598', '+17·2'],
  ['PseudoLabel', '0·405 (0·358–0·449)', '0·610 (0·574–0·645)', '0·600', '+20·6'],
  ['SimCLR', '0·446 (0·397–0·499)', '0·602 (0·557–0·645)', '0·582', '+15·6'],
  ['DANN', '0·396 (0·352–0·440)', '0·590 (0·559–0·621)', '0·580', '+19·4'],
  ['Two-layer perceptron', '0·400 (0·359–0·443)', '0·412 (0·370–0·453)', '0·405', '+1·1'],
  ['Deep perceptron', '0·398 (0·357–0·438)', '0·401 (0·363–0·438)', '0·396', '+0·3'],
  ['EEG Conformer', '0·353 (0·321–0·385)', '0·361 (0·330–0·394)', '0·354', '+0·8'],
];
add(
  new Paragraph({ children: runs('**Table 1: Cross-participant decoding of four emotions in SEED-IV, zero-shot and after 20-window calibration**', { size: 20 }), spacing: { before: 200, after: 100 } }),
  table(['Model', 'Zero-shot balanced accuracy', 'Calibrated balanced accuracy', 'Calibrated macro F1', 'Calibration gain, points'], T1, [2226, 1950, 1950, 1450, 1450], { highlightRow: (r) => r[1] === '' }),
  Caption('Values are means over 15 held-out participants (each averaged over three seeds), with 95% bootstrap CIs. Chance balanced accuracy is 0·25. Calibration fine-tuned the network on 20 labelled windows (about 80 s) of the held-out participant, sampled across trials (see Methods for the resulting optimism).'),
);
add(H2('Zero-shot decoding'));
add(
  P('Without target data, no model approached the accuracies reported in much of the literature (figure 1). With 62 electrodes, CLISA-EWU had the highest mean zero-shot balanced accuracy (0·485, 95% CI 0·428–0·542), but it did not differ from the two-layer perceptron (difference 0·9 percentage points, –0·5 to 2·1; Wilcoxon p=0·25) or the deep perceptron (0·6 points, –0·9 to 2·2; p=0·64). Its advantage over DANN (2·7 points, 1·0 to 4·2; unadjusted p=0·010) did not survive Holm correction (p=0·12). With six electrodes, zero-shot accuracy fell to 0·406 (0·370–0·439) for CLISA-EWU, and SimCLR (0·446) and BYOL (0·440) were numerically higher (both Holm-adjusted p=1·00). Zero-shot performance varied greatly between participants, from 0·244 (participant 11, at chance) to 0·723 (participant 7) for CLISA-EWU with 62 electrodes (figure 2A).'),
);
add(Img(`${FIG}/figure1_zeroshot_vs_calibrated.png`, 600, 367));
add(Caption('**Figure 1: Zero-shot versus calibrated balanced accuracy for 16 deep models** (A) 62 electrodes. (B) Six electrodes (FP1, FP2, F7, F8, T7, T8). Points are means over 15 held-out participants; bars are 95% bootstrap CIs; the grey segment joins the two protocols for each model. The dotted line marks chance (0·25).'));
add(H2('Effect of few-shot calibration'));
add(
  P('Calibration with 20 labelled windows changed the ranking of models (figure 1; table 1). Models trained with a contrastive, domain-adversarial, or pseudo-label objective (CLISA-EWU, SimCLR, BYOL, DANN, and PseudoLabel) gained 15·6–21·4 percentage points (MixMatch and the DANCE pair gained 9·4–25·0 points), whereas purely supervised networks gained 0·1–2·0 points under the identical calibration routine (appendix table S2). After calibration, CLISA-EWU had the highest mean balanced accuracy with both montages: 0·678 (0·641–0·718) with 62 electrodes and 0·620 (0·590–0·648) with six. The mechanism behind the objective-dependent gain is not established by this study; candidate explanations include adaptation of batch-normalisation statistics to the new participant^19^ and differences in how sharply each training objective fits the training participants. [[Pending ablation: calibrate (i) batch-normalisation statistics only, (ii) the classifier only, and (iii) all parameters, for CLISA-EWU and the two-layer perceptron.]]'),
);
const T2 = [
  ['62 electrodes', '', '', '', ''],
  ['vs DANN', '1·6 (0·6 to 2·6)', '11/15', '0·013', '0·062'],
  ['vs PseudoLabel', '2·1 (1·0 to 3·1)', '13/15', '0·003', '0·016'],
  ['vs BYOL', '6·9 (5·2 to 8·5)', '15/15', '<0·001', '0·001'],
  ['vs SimCLR', '10·3 (8·4 to 12·2)', '15/15', '<0·001', '0·001'],
  ['vs two-layer perceptron', '18·2 (14·4 to 22·0)', '15/15', '<0·001', '0·001'],
  ['Six electrodes', '', '', '', ''],
  ['vs DANN', '3·0 (0·9 to 5·0)', '11/15', '0·022', '0·086'],
  ['vs PseudoLabel', '1·0 (–0·7 to 2·7)', '9/15', '0·33', '0·69'],
  ['vs BYOL', '0·8 (–1·8 to 3·3)', '8/15', '0·68', '0·69'],
  ['vs SimCLR', '1·8 (–1·3 to 4·7)', '10/15', '0·23', '0·69'],
  ['vs two-layer perceptron', '20·8 (17·0 to 24·9)', '15/15', '<0·001', '0·001'],
];
add(
  new Paragraph({ children: runs('**Table 2: Paired participant-level comparisons of CLISA-EWU with other models after calibration**', { size: 20 }), spacing: { before: 200, after: 100 } }),
  table(['Comparison', 'Difference in balanced accuracy, points (95% CI)', 'Participants favouring CLISA-EWU', 'Wilcoxon p', 'Holm-adjusted p'], T2, [2426, 2300, 1600, 1350, 1350], { highlightRow: (r) => r[1] === '' }),
  Caption('Differences are CLISA-EWU minus comparator, averaged over 15 held-out participants. Holm adjustment covered 14 comparisons (seven comparators × two montages; deep perceptron and EEG Conformer comparisons, all p<0·001, are in appendix table S4).'),
);
add(
  P('In paired participant-level comparisons (table 2), calibrated CLISA-EWU outperformed BYOL, SimCLR, and all supervised networks with 62 electrodes, for every one of the 15 participants. Its margin over PseudoLabel (2·1 points) remained significant after correction, but its margin over DANN (1·6 points) did not. With six electrodes, CLISA-EWU, BYOL, PseudoLabel, and SimCLR were statistically indistinguishable.'),
);
add(H2('Electrode reduction'));
add(
  P('Reducing the montage from 62 to six electrodes lowered calibrated balanced accuracy for CLISA-EWU by 5·8 percentage points (Wilcoxon p<0·001), a retention of 91·4% (95% CI 87·9–95·0; figure 2B). Zero-shot retention was lower, at 83·6%. Retention differed between models (appendix table S5): BYOL and SimCLR performed as well or better with six electrodes than with 62 whereas EEG Conformer retained 78·5%. After calibration, every six-electrode contrastive, domain-adversarial, and pseudo-label model remained well above chance for every participant (CLISA-EWU minimum 0·518).'),
);
add(Img(`${FIG}/figure2_per_participant_montage.png`, 600, 283));
add(Caption('**Figure 2: Participant-level balanced accuracy of CLISA-EWU with 62 and six electrodes** (A) Zero-shot. (B) After 20-window calibration. Each line is one held-out participant (mean of three seeds); black bars are means over participants. The dotted line marks chance (0·25).'));
add(H2('Variability and robustness'));
add(
  P('Between-participant variability exceeded between-seed variability. With 62 electrodes, calibrated CLISA-EWU ranged from 0·541 (participant 5) to 0·813 (participant 2) across participants, whereas seed means ranged from 0·658 to 0·691. The highest single run (0·854; participant 15, seed 7) is reported in appendix table S6 only for completeness, because a best fold is not an estimate of expected performance. Per-class recall, confusion matrices, and representation-geometry analyses restricted to held-out participants are reported in the appendix. [[Recompute clustering metrics using only the held-out participant’s windows; the current values were computed on all 15 participants, 14 of whom were in the training set.]]'),
);
add(H2('External validation'));
add(P('[[Pending. Repeat the full protocol on SEED, SEED-V, and FACED (using the same six-electrode positions where available), and train on SEED-IV and test on the other datasets without retraining. Report table 3 in the same format as table 1.]]'));
add(
  new Paragraph({ children: runs('**Table 3: External validation** [[pending]]', { size: 20 }), spacing: { before: 200, after: 100 } }),
  table(['Dataset (participants, classes)', 'Zero-shot', 'Calibrated', 'Six-electrode retention'], [
    ['SEED (15, 3)', '[[ ]]', '[[ ]]', '[[ ]]'], ['SEED-V (16, 5)', '[[ ]]', '[[ ]]', '[[ ]]'], ['FACED (123, 9 or 3)', '[[ ]]', '[[ ]]', '[[ ]]'], ['Wearable recording, Bangladesh cohort (n=[[ ]])', '[[ ]]', '[[ ]]', 'not applicable'],
  ], [3626, 1800, 1800, 1800]),
);

// ---------- Discussion ----------
add(H1('Discussion'));
add(
  P('In this benchmark of 26 models evaluated under one leave-one-participant-out protocol, three findings stand out. First, decoding four emotions in an unseen participant without any of their data was modest for every model, and the deep architectures, including the contrastive model of primary interest, did not clearly outperform a two-layer perceptron. Second, about 80 s of labelled calibration data raised balanced accuracy by roughly 20 percentage points, but only for models trained with contrastive, domain-adversarial, or pseudo-label objectives. Third, a six-electrode frontal–temporal montage retained about nine-tenths of calibrated full-montage performance.'),
  P('These findings change how cross-participant results in this field should be read. Much of the published progress on SEED-IV has been attributed to architectural innovations, yet our results suggest that the protocol, and specifically whether and how target-participant data are used, may explain as much variance as the architecture. Under calibration, the ordering of model families inverted: supervised networks that matched or beat self-supervised models zero-shot fell far behind them. Comparisons that mix zero-shot, transductive, and calibrated results, or that compare a best fold with another study’s mean, are therefore uninterpretable. We recommend that cross-participant EEG studies report zero-shot and calibrated results separately, specify the number and selection of calibration samples, keep calibration and test trials disjoint, and summarise at the participant level with CIs, in line with TRIPOD+AI.^17,24^'),
  P('The practical message for digital health is cautiously positive. A short calibration session is acceptable in most consumer and clinical workflows, and a headband carrying frontal and temporal electrodes appears to keep most of the decodable signal under calibration. The fact that some self-supervised models lost nothing from electrode reduction suggests that the frontal–temporal subset contains much of the information these models exploit, consistent with the established role of frontal asymmetry in affect.^1,3^ However, the absolute calibrated accuracy (about 62% for four classes with six electrodes) and the large between-participant spread mean that individual predictions remain unreliable. The system is not suitable for diagnosis or for clinical decisions.'),
  P('This study has limitations. The cohort was small (15 participants), healthy, young, and drawn from one country and laboratory, and emotions were elicited with film clips under controlled conditions; generalisation to other populations, naturalistic settings, and clinical groups such as people with depression is unknown. [[External validation results will partly address this.]] The six-electrode montage was simulated by selecting channels from a laboratory cap rather than recorded with a wearable device, which would add motion artefacts, dry-electrode noise, and different referencing. [[Report the pilot wearable recording when available.]] Calibration windows were sampled across trials, so calibration and test windows could share a trial, which, with temporally smoothed features, may have inflated calibrated accuracy;^25^ the trial-held-out analysis will quantify this. Labels were assigned per film clip rather than per moment, and self-reported intensity was not modelled. We used precomputed features, so preprocessing choices could not be varied. Hyperparameters were not tuned separately for each model, which may disadvantage some comparators. Finally, our contrastive model contrasted augmented views of single windows rather than aligning participants explicitly, so its gains should not be attributed to inter-participant alignment.'),
  P('In conclusion, in a controlled participant-independent benchmark, brief labelled calibration rather than architecture accounted for most of the measurable gain in EEG emotion decoding, and a six-electrode wearable-compatible montage retained about 91% of calibrated performance. Rigorous separation of zero-shot and calibrated evaluation, external validation, and real wearable data are needed before EEG-based affect sensing can support digital health applications.'),
);

// ---------- Back matter ----------
add(H1('Contributors'));
add(P('SK conceived the study, developed the models, ran the experiments, and drafted the manuscript. OBQ contributed to experiments, analysis, and manuscript preparation. KMFAB contributed to implementation, evaluation, and documentation. RR contributed to experiments, reporting, and software integration. AWR supervised the project and critically revised the manuscript. SK and [[name]] accessed and verified the underlying data. All authors had full access to all the data, approved the final version, and accepted responsibility for the decision to submit for publication.'));
add(H1('Declaration of interests'));
add(P('We declare no competing interests.'));
add(H1('Data sharing'));
add(P('SEED-IV, SEED, SEED-V, and FACED are available from their providers under their licences. All code, data splits, per-fold predictions, confusion matrices, and the analysis script used to produce the tables and figures are available at https://github.com/saifkhancse/clisa-ewu-brain-informatics. [[Archive a tagged release on Zenodo and cite its DOI; release model weights.]]'));
add(H1('Acknowledgments'));
add(P('We thank the providers of the SEED-IV dataset and the Department of Computer Science and Engineering, East West University.'));

add(H1('References'));
const refs = [
  'Russell JA. A circumplex model of affect. J Pers Soc Psychol 1980; 39: 1161–78.',
  'Zheng WL, Liu W, Lu Y, Lu BL, Cichocki A. EmotionMeter: a multimodal framework for recognizing human emotions. IEEE Trans Cybern 2019; 49: 1110–22.',
  'Zheng WL, Lu BL. Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks. IEEE Trans Auton Ment Dev 2015; 7: 162–75.',
  'Shen X, Liu X, Hu X, Zhang D, Song S. Contrastive learning of subject-invariant EEG representations for cross-subject emotion recognition. IEEE Trans Affect Comput 2022; published online April 1. https://doi.org/10.1109/TAFFC.2022.3164516. [[verify author list]]',
  'Ganin Y, Ustinova E, Ajakan H, et al. Domain-adversarial training of neural networks. J Mach Learn Res 2016; 17: 1–35.',
  'Koelstra S, Muhl C, Soleymani M, et al. DEAP: a database for emotion analysis using physiological signals. IEEE Trans Affect Comput 2012; 3: 18–31.',
  'Li H, Jin YM, Zheng WL, Lu BL. Cross-subject emotion recognition using deep adaptation networks. In: Neural Information Processing (ICONIP 2018). Cham: Springer, 2018: 403–13.',
  'Li Y, Zheng W, Cui Z, Zhang T, Zong Y. A novel neural network model based on cerebral hemispheric asymmetry for EEG emotion recognition. Proceedings of IJCAI 2018: 1561–67.',
  'Li Y, Wang L, Zheng W, et al. A novel bi-hemispheric discrepancy model for EEG emotion recognition. IEEE Trans Cogn Dev Syst 2021; 13: 354–67.',
  'Zhong P, Wang D, Miao C. EEG-based emotion recognition using regularized graph neural networks. IEEE Trans Affect Comput 2022; 13: 1290–301.',
  'Li J, Li S, Pan J, Wang F. Cross-subject EEG emotion recognition with self-organized graph neural network. Front Neurosci 2021; 15: 611653.',
  'Chen T, Kornblith S, Norouzi M, Hinton G. A simple framework for contrastive learning of visual representations. Proceedings of ICML 2020: 1597–607.',
  'Grill JB, Strub F, Altché F, et al. Bootstrap your own latent: a new approach to self-supervised learning. Adv Neural Inf Process Syst 2020; 33: 21271–84.',
  'Lee DH. Pseudo-label: the simple and efficient semi-supervised learning method for deep neural networks. ICML Workshop on Challenges in Representation Learning, 2013.',
  'Berthelot D, Carlini N, Goodfellow I, Papernot N, Oliver A, Raffel C. MixMatch: a holistic approach to semi-supervised learning. Adv Neural Inf Process Syst 2019; 32.',
  'Song Y, Zheng Q, Liu B, Gao X. EEG Conformer: convolutional transformer for EEG decoding and visualization. IEEE Trans Neural Syst Rehabil Eng 2023; 31: 710–19.',
  'Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024; 385: e078378.',
  'Chen J, Wang X, Huang C, Hu X, Shen X, Zhang D. A large finer-grained affective computing EEG dataset. Sci Data 2023; 10: 740.',
  'Li Y, Wang N, Shi J, Hou X, Liu J. Adaptive batch normalization for practical domain adaptation. Pattern Recognit 2018; 80: 109–17.',
  'Song T, Zheng W, Song P, Cui Z. EEG emotion recognition using dynamical graph convolutional neural networks. IEEE Trans Affect Comput 2020; 11: 532–41.',
  'Hinton G, Vinyals O, Dean J. Distilling the knowledge in a neural network. arXiv 2015; published online March 9. https://arxiv.org/abs/1503.02531.',
  'Liu W, Qiu JL, Zheng WL, Lu BL. Comparing recognition performance and robustness of multimodal deep learning models for multimodal emotion recognition. IEEE Trans Cogn Dev Syst 2022; 14: 715–29.',
  'Saeb S, Lonini L, Jayaraman A, Mohr DC, Kording KP. The need to approximate the use-case in clinical machine learning. GigaScience 2017; 6: 1–9.',
  'Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns 2023; 4: 100804.',
  'Li R, Johansen JS, Ahmed H, et al. The perils and pitfalls of block design for EEG classification experiments. IEEE Trans Pattern Anal Mach Intell 2021; 43: 316–33.',
];
refs.forEach((r) => add(new Paragraph({ children: runs(r, { size: 20 }), numbering: { reference: 'refs', level: 0 }, spacing: { after: 60, line: 280 } })));
add(P('[[Reference note: every reference was carried over from the thesis or is a standard citation, but all bibliographic details (authors, pages, years) must be checked against the source before submission. Renumber in order of first citation after final edits; citations 6, 20 and 21 are currently cited only in the appendix.]]', { run: { size: 18 } }));

const doc = new Document({
  styles, numbering,
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } }, lineNumbers: { countBy: 1, restart: d.LineNumberRestartFormat.CONTINUOUS } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log('written', OUT); });
