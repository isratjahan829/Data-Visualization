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

// ---- Section structure: Introduction / Literature Review / Materials and Methods / Results and Discussion / Conclusion and Future Work ----
let s1 = 0, s2 = 0;
const S1 = (t) => { s1++; s2 = 0; H1(`${s1} ${t}`); };
const S2 = (t) => { s2++; H2(`${s1}.${s2} ${t}`); };
const MIX = (arr) => arr.forEach(([t, x]) => (t === 'eq' ? EQ(x) : P(x)));
const BUL = (text) => B.push({ t: 'bul', text });

const ORG2 = 'The remainder of this paper is organized as follows. Section 2 reviews the literature on cross-participant and reduced-electrode EEG emotion recognition. Section 3 describes the materials and methods: the data, preprocessing and augmentation, the evaluation protocol, the baseline and proposed models, the training algorithm, the evaluation metrics and the statistical analysis. Section 4 presents and discusses the results, including their implications and limitations. Section 5 concludes the paper and outlines future work.';
const LIT_INTRO = 'Published EEG emotion recognition methods can be grouped into domain-adaptive networks, graph and attention models, contrastive and self-supervised methods, and reduced-electrode approaches. This section reviews each group, summarises the reported results in Table {T:lit} and identifies the research gap addressed by this study.';
const GAP = 'Three gaps emerge from this literature. First, reported cross-participant accuracies mix zero-shot, transductive and calibrated protocols and are often summarised as a best fold rather than a participant mean, so they cannot be compared. Second, no study has separated, under one protocol, how much of the performance of classical, supervised, self-supervised, adversarial and semi-supervised models comes from labelled data of the new user. Third, electrode reduction is rarely quantified with the same models, the same protocol and participant-level uncertainty. The present study addresses all three.';
const DATA_DESC = 'The public SEED-IV dataset{seediv} was used. Fifteen healthy adults [[report sex and age range from the dataset paper]] watched 72 film clips (24 per session, three sessions on different days) selected to elicit neutral, sad, fear and happy states while 62-channel EEG was recorded according to the international 10–20 system. The released feature set contains 37,575 four-second windows, 2505 per participant, with 10,170 neutral (27.1%), 10,245 sad (27.3%), 9225 fear (24.6%) and 7935 happy (21.1%) windows (Fig. {F:dataset}); the classes are therefore close to balanced. Labels are assigned per film clip, so all windows of a trial share one label.';
const PREPROC = [
  'The EEG in SEED-IV was downsampled to 200 Hz and band-pass filtered by the dataset providers{seediv,seed}. Differential entropy (DE) features were computed in non-overlapping 4-s windows for five frequency bands (delta, theta, alpha, beta and gamma) [[confirm band limits against the dataset documentation]]. For a band-limited signal that is approximately Gaussian with variance σ², the differential entropy reduces to a function of band power:',
  'The released DE features were smoothed over time with a linear dynamical system (de_LDS){seed}, which reduces window-to-window noise but also increases the correlation between neighbouring windows of a trial. For each window the 62 × 5 DE matrix was flattened into a 310-dimensional vector. Each feature *j* was z-standardised,',
  'where μ~j~ and σ~j~ are the mean and standard deviation of feature *j* [[confirm that μ and σ are computed on the training participants of each fold; if they were computed on all 15 participants, re-run with fold-wise statistics]]. No further normalisation was applied.',
];
const MONTAGE = 'Two electrode configurations were compared. The full montage used all 62 electrodes (310 features). The wearable-compatible montage used six electrodes reachable by a forehead-and-temple band, FP1, FP2, F7, F8, T7 and T8, giving 30 features (6 electrodes × 5 bands). The six-electrode features are an exact column subset of the full feature matrix (indices 0, 2, 5, 13, 23 and 31 of the 62-channel ordering), so the two configurations differ only in electrode density and not in preprocessing.';
const AUGMENT = [
  'Data augmentation was used to create the two views required by contrastive pretraining. For a standardised window *x*, the first view adds isotropic Gaussian noise, and the second view masks whole electrodes across all five bands with a random binary mask *m*, each electrode being kept with probability 0.7:',
  'The noise view encourages invariance to small amplitude perturbations, and the masking view encourages invariance to missing or poorly contacting electrodes, a frequent problem with wearable headsets. Augmentation was applied on the fly to each mini-batch during stage 1 only; supervised fine-tuning and evaluation used unaugmented windows. To counter the mild class imbalance, training mini-batches were drawn with a class-balanced sampler whose sampling weight for a window of class *i* is proportional to 1/*n*~i~.',
];
const MODEL_INTRO = 'CLISA-EWU (Fig. {F:arch}) consists of an encoder *f*~θ~, a projection head *g*~φ~ used only during pretraining, and a linear classifier. The encoder maps a window *x* to a 128-dimensional representation *h*, and the projection head maps *h* to a 64-dimensional unit vector *z*:';
const MODEL_LAYERS = 'where *W*~1~ ∈ ℝ^256×d^ and *W*~2~ ∈ ℝ^128×256^ (*d* = 310 or 30), BN denotes batch normalisation and dropout has rate 0.3. The classifier outputs class probabilities ŷ = softmax(*W*~c~*h* + *b*~c~). Training has two stages. In stage 1, the encoder and projection head are trained with the NT-Xent contrastive loss{simclr} on the two augmented views. For a batch of *N* windows, the views give 2*N* projections, and for a positive pair (*i*, *j*) the loss is';
const MODEL_STAGE2 = 'where sim(·,·) is cosine similarity and τ = 0.5. In stage 2, the encoder and classifier are fine-tuned with the cross-entropy loss over mini-batches B,';
const MODEL_NOTE = 'Unlike the original CLISA{clisa}, which treats time-aligned responses of different participants to the same stimulus as positives, the stage-1 objective here contrasts augmented views of the same window and does not use participant or stimulus identity [[either implement cross-participant stimulus-aligned positives or rename the model to avoid implying explicit inter-subject alignment]].';
const ALGO_INTRO = 'Algorithm {A:1} summarises the complete training and evaluation procedure for one montage, including both evaluation protocols.';
const FUTURE = [
  'External validation on SEED, SEED-V and FACED, and training on SEED-IV with testing on these datasets without retraining{seed,seedv,faced}.',
  'A trial-held-out calibration protocol (calibrate on session 1, test on sessions 2–3) and a calibration-budget curve of 5–200 windows to remove the trial-overlap optimism of the present calibration.',
  'An ablation that calibrates only batch-normalisation statistics, only the classifier, or all parameters, to identify the mechanism behind the objective-dependent calibration gain.',
  'A true inter-participant contrastive objective with stimulus-aligned positives from different participants, to test whether zero-shot accuracy can be improved.',
  'A seven- or eight-electrode montage that adds a parieto-occipital site, motivated by the attribution results for fear.',
  'Recording a new cohort with a dry-electrode wearable headband and real-time deployment of the NeuroSync interface, followed by studies in clinically relevant populations.',
];

// ---- Expanded literature review text ----
const LIT_INTRO2 = 'This section reviews the work on which the present study builds. It moves from the psychological and neurophysiological basis of emotion recognition, through datasets, features and deep architectures, to the transfer-learning, self-supervised and evaluation practices that determine whether a decoder generalises to new people, and finally to reduced-electrode and wearable EEG. Reported results are summarised in Table {T:lit}, the main method families are contrasted in Table {T:families}, and the section closes with the research gap addressed by this study.';
const LIT_AFFECT = [
  'Computational studies of emotion usually adopt either discrete categories, such as the neutral, sad, fear and happy states of SEED-IV, or dimensional models in which emotions are located in a space spanned by valence and arousal{russell}. The discrete view simplifies labelling with film clips, whereas the dimensional view underlies datasets such as DEAP{deap} and DREAMER{dreamer}. Both rest on the observation that affective states modulate cortical oscillations. The best-established correlate is frontal alpha asymmetry, in which relatively greater left frontal activity accompanies approach-related, positive affect and greater right frontal activity accompanies withdrawal-related, negative affect{davidson,coan}. Beyond the frontal lobes, emotion-related changes have been reported across theta, beta and gamma bands and in temporal and parietal regions, and high-frequency beta and gamma activity over lateral temporal sites has been found to be particularly discriminative for emotion{seed}.',
  'Systematic surveys of EEG emotion recognition{alarcao,jenke} highlight two consequences of this neurophysiology for machine learning. First, informative features are spread over many frequency bands and electrodes, which motivates multichannel spectral features and spatial models. Second, the same emotional stimulus evokes different spectral patterns in different people, so that a model trained on one group generalises poorly to another; this inter-participant variability is the central difficulty that the present study examines.',
];
const LIT_DATA = [
  'Public datasets have made controlled comparison possible. DEAP recorded 32-channel EEG from 32 participants watching music videos rated for valence and arousal{deap}; SEED recorded 62-channel EEG from 15 participants watching film clips eliciting positive, neutral and negative emotions{seed}; SEED-IV extended this design to four discrete emotions over three sessions{seediv}; SEED-V added a fifth class{seedv}; DREAMER used a 14-channel consumer headset{dreamer}; and FACED provides 32-channel recordings from 123 participants with nine fine-grained emotion categories{faced}. Because SEED and SEED-IV release precomputed differential-entropy features, they have become the de facto benchmarks for cross-participant studies.',
  'In these datasets, each film clip yields many consecutive windows that share a label. This structure creates two distinct generalisation problems: generalisation to new windows or trials of a participant whose other data were used for training (subject-dependent), and generalisation to a participant whose data were never seen (subject-independent or cross-participant). Accuracies for the second problem are consistently and substantially lower than for the first{jayaram,wu}, and it is the second that matters for deployment.',
];
const LIT_FEAT = [
  'Early work relied on hand-crafted features. Jenke and colleagues compared a broad set of time-domain, frequency-domain and time–frequency features and found that band-power-related features and their combinations were among the most informative{jenke}. Differential entropy (DE), the entropy of a band-limited signal, which for approximately Gaussian signals equals the logarithm of band power up to a constant, was introduced as a robust emotion feature{duan} and outperformed band power in SEED{seed}. Smoothing the DE sequence with a linear dynamical system further improved accuracy by suppressing window-to-window noise{seed}, at the cost of introducing temporal dependence between neighbouring windows. Classical classifiers such as support vector machines, linear discriminant analysis and ensembles of decision trees remain strong baselines on these features{lotte,xgboost}, and our own classical benchmark confirms that they carry emotion information across participants.',
];
const LIT_DEEP = [
  'Deep learning has been applied both to raw EEG and to spectral features{craik,roy}. Convolutional architectures designed for raw EEG, such as the deep and shallow ConvNets of Schirrmeister and colleagues{schirrmeister} and the compact EEGNet{eegnet}, learn temporal and spatial filters end to end, and transformer-based models such as EEG Conformer add global self-attention{conformer,vaswani}. For DE features, which are organised by electrode and band, graph neural networks have been particularly influential. DGCNN learns an adjacency matrix between electrodes{dgcnn}; RGNN regularises this graph with anatomical connectivity and node-wise domain adversarial training{rgnn}; SOGNN constructs a sample-specific graph{sognn}; and hierarchical spatial–temporal models aggregate regional and global brain information{regional}. More recent attention-based and frequency-band graph models{attgraph,bfenet} and self-supervised graph models{ssemerge} report the highest SEED-IV accuracies (Table {T:lit}).',
  'Two observations temper these advances. First, systematic reviews have noted that the reported gains of deep models over classical pipelines vary widely, depend strongly on the evaluation protocol and are often obtained with within-participant evaluation{craik,roy}. Second, the architectural gains reported on SEED-IV are entangled with differences in evaluation protocol, as discussed below, so that the contribution of the architecture itself is difficult to isolate.',
];
const LIT_DA = [
  'Because inter-participant variability resembles a shift between data distributions, cross-participant EEG has been framed as a transfer-learning or domain-adaptation problem{jayaram,wu}. Shallow approaches align feature distributions directly, for example with transfer component analysis{pan}, and early EEG studies showed that personalising models through transfer learning improves accuracy on new participants{zhengpers}. In brain–computer interfaces more broadly, Riemannian re-centring of covariance matrices{zanini} and Euclidean alignment of each participant’s data{he} reduce inter-participant shift with little or no labelled target data.',
  'Deep domain adaptation extends these ideas to learned representations. Deep adaptation networks minimise a maximum mean discrepancy between source and target features{long}, and their application to SEED and SEED-IV improved cross-participant accuracy{dan}. Domain-adversarial training removes domain information with a gradient-reversal layer{ganin} and has been combined with bi-hemispheric models{bidann,bihdm}. Multisource transfer selects and weights the most similar source participants{multisource}, and plug-and-play domain adaptation adapts a pretrained model to a new participant with a small amount of target data{plugplay}. A simpler mechanism, adaptive batch normalisation, re-estimates normalisation statistics on target data without changing the learned weights{adabn}. A common feature of most of these methods is that they use data from the test participant: unlabelled data in the transductive setting, or a small labelled set in the calibration setting. Their reported accuracies therefore measure a different quantity from zero-shot generalisation, a distinction that is rarely made explicit.',
];
const LIT_SSL = [
  'Self-supervised learning offers a way to learn representations without labels. Contrastive methods such as SimCLR pull together two augmented views of the same sample and push apart other samples{simclr}, while BYOL avoids negative pairs by predicting one view from another with a slowly updated target network{byol}. For EEG, contrastive pretraining with signal-specific augmentations{mohsenvand}, relative-positioning and temporal-shuffling pretext tasks{banville} and transformer-based contrastive pretraining on large unlabelled corpora{bendr} have all produced representations that transfer to downstream tasks. CLISA adapted contrastive learning specifically to cross-participant emotion recognition by treating the time-aligned responses of different participants to the same stimulus as positive pairs, explicitly encouraging inter-participant alignment{clisa}.',
  'Semi-supervised methods exploit unlabelled data during supervised training. Pseudo-labelling adds confident predictions on unlabelled data as training targets{pseudo}, and MixMatch combines consistency regularisation, entropy minimisation and mixing{mixmatch}. When the unlabelled data come from the test participant, these methods become transductive. Knowledge distillation{hinton} has been used to transfer information from a large teacher to a compact student and is relevant when moving from dense to sparse electrode montages.',
];
const LIT_EVAL = [
  'In motor-imagery and event-related brain–computer interfaces, a short calibration session for each new user is standard practice, and transfer learning is primarily used to shorten it{lotte,jayaram,wu}. In EEG emotion recognition, by contrast, results labelled as cross-participant or subject-independent may be zero-shot, transductive or calibrated, and the protocol is often described only implicitly. The amount of target data and how it is selected are rarely reported, even though, as the present study shows, they can dominate the outcome.',
  'Data splitting raises related concerns. When windows from the same participant or trial appear in both training and test data, models can exploit participant identity or temporal autocorrelation rather than the effect of interest, inflating accuracy{saeb,kapoor}. In EEG, block designs in which all samples of a class are recorded together allow classifiers to learn slow temporal drifts instead of class-specific responses{liblock}. Finally, many studies report the mean of a small number of runs without participant-level uncertainty, and some report the best fold, which is not an estimate of expected performance. Reporting guidelines for prediction models, such as TRIPOD+AI{tripod}, and statistical procedures for comparing classifiers across data sets{demsar} provide a framework for addressing these issues that has seldom been applied to EEG emotion recognition.',
];
const LIT_WEAR = [
  'Practical affective interfaces require few electrodes. Consumer headsets such as the 14-channel device used in DREAMER{dreamer} and four-channel forehead-and-ear headbands have been validated for some research purposes{krigolson}, and wearable EEG is an active area of engineering development{casson}. In SEED, Zheng and Lu showed that selected subsets of four to six lateral temporal electrodes retained accuracy close to the full montage in subject-dependent classification{seed}, consistent with the emotion-related information carried by frontal and temporal sites{davidson,coan}. Knowledge distillation from dense to sparse montages has also been proposed{hinton}.',
  'However, most reduced-electrode studies evaluate subject-dependent performance, select electrodes using data from the same participants on which they are tested, or compare different models across montages. How much cross-participant performance survives electrode reduction when the same models and the same protocol are used, and how this depends on calibration, has not been quantified with participant-level uncertainty.',
];
const GAP2 = 'Four gaps emerge from this literature. First, reported cross-participant accuracies mix zero-shot, transductive and calibrated protocols and are sometimes summarised as a best fold rather than a participant mean, so they cannot be compared (Table {T:lit}). Second, no study has separated, under one protocol, how much of the performance of classical, supervised, self-supervised, adversarial and semi-supervised models (Table {T:families}) comes from labelled data of the new user. Third, the consequences of temporal dependence between windows for calibration and validation are rarely considered. Fourth, electrode reduction is rarely quantified with the same models, the same protocol and participant-level uncertainty. The present study addresses these gaps with a 26-model benchmark that reports zero-shot and calibrated performance separately, uses participant-level statistics, documents the limitations of its own calibration protocol, and evaluates a six-electrode wearable-compatible montage under identical conditions.';
const FAMILIES = { title: 'Main families of cross-participant EEG emotion recognition methods, the target-participant data they typically use, and the main limitation relevant to this study.', header: ['Family', 'Representative work', 'Target data typically used', 'Main limitation'], rows: [
  ['Hand-crafted features + classical classifiers', 'DE features{duan,seed}; feature surveys{jenke}', 'None (zero-shot)', 'Limited capacity to model inter-participant variability'],
  ['Graph and attention networks', 'DGCNN{dgcnn}, RGNN{rgnn}, SOGNN{sognn}, AttGraph{attgraph}', 'Varies; often unlabelled target data', 'Protocol and reporting unit often unclear'],
  ['Shallow transfer and alignment', 'TCA{pan}, Riemannian{zanini} and Euclidean{he} alignment', 'Unlabelled target data', 'Assumes a stable target distribution'],
  ['Deep domain adaptation', 'DAN{dan}, DANN{ganin}, BiDANN{bidann}, plug-and-play DA{plugplay}', 'Unlabelled or a few labelled target samples', 'Transductive; not zero-shot'],
  ['Self-supervised and contrastive', 'SimCLR{simclr}, BYOL{byol}, BENDR{bendr}, CLISA{clisa}', 'None for pretraining; labels for fine-tuning', 'Gains often reported after fine-tuning or calibration'],
  ['Semi-supervised', 'Pseudo-label{pseudo}, MixMatch{mixmatch}', 'Unlabelled target data when transductive', 'Transductive; confirmation bias'],
  ['Reduced-electrode and distillation', 'Channel selection{seed}, distillation{hinton}', 'Varies', 'Mostly subject-dependent evaluation'],
], rel: [22, 30, 22, 26] };

TABLES.families = FAMILIES; { const tot = FAMILIES.rel.reduce((a, b) => a + b, 0); FAMILIES.widths = FAMILIES.rel.map((r) => Math.floor((r * TEXT_W) / tot)); FAMILIES.widths[FAMILIES.widths.length - 1] += TEXT_W - FAMILIES.widths.reduce((a, b) => a + b, 0); }

// ---- Blocks ----
S1('Introduction');
C.INTRO.forEach((x) => P(x.replace(' (Table {T:lit})', '')));
P('This work addresses these gaps through a controlled benchmark with a rigorous, participant-level evaluation. The main contributions of this research are:');
CONTRIB_SR.forEach(([a, b]) => B.push({ t: 'num', text: `*${a}*: ${b}` }));
P(ORG2);

S1('Literature Review'); P(LIT_INTRO2);
S2('Emotion models and EEG correlates of affect'); LIT_AFFECT.forEach(P);
S2('Datasets and the cross-participant problem'); LIT_DATA.forEach(P);
S2('Hand-crafted features and classical classifiers'); LIT_FEAT.forEach(P);
S2('Deep architectures for EEG emotion recognition'); LIT_DEEP.forEach(P);
S2('Transfer learning and domain adaptation across participants'); LIT_DA.forEach(P);
S2('Self-supervised, contrastive and semi-supervised learning'); LIT_SSL.forEach(P);
S2('Calibration, evaluation protocols and reporting'); LIT_EVAL.forEach(P);
S2('Reduced-electrode and wearable EEG'); LIT_WEAR.forEach(P);
S2('Research gap'); P(GAP2);

S1('Materials and Methods'); P(METH_INTRO);
S2('Data description'); P(DATA_DESC); P(ETHICS);
S2('Data preprocessing and feature extraction'); P(PREPROC[0]); EQ('de'); P(PREPROC[1]); EQ('zscore'); P(PREPROC[2]);
S2('Electrode montages'); P(MONTAGE);
S2('Data augmentation and class balancing'); P(AUGMENT[0]); EQ('aug'); P(AUGMENT[1]);
S2('Data splitting and evaluation protocol'); P(C.M_PROTOCOL[0]);
H3('Strategy 1: zero-shot evaluation'); P(STRAT1);
H3('Strategy 2: few-shot calibration'); P(STRAT2); P(CAVEAT);
S2('Baseline models'); C.M_MODELS.forEach(P);
S2('Proposed model: CLISA-EWU'); P(MODEL_INTRO); EQ('encoder'); P(MODEL_LAYERS); EQ('ntxent'); P(MODEL_STAGE2); EQ('ce'); P(MODEL_NOTE);
S2('Training algorithm'); P(ALGO_INTRO); B.push({ t: 'algo' });
S2('Implementation details'); P(TRAINCFG); P(IMPL);
S2('Evaluation metrics'); MIX(C.METRICS);
S2('Statistical analysis'); MIX(C.STATS);
S2('Representation and attribution analysis'); P(C.M_ANALYSIS[0]);

S1('Results and Discussion'); P(RD_INTRO);
S2('Benchmark overview'); P(C.R_OVERVIEW.replace('The experimental workflow and the CLISA-EWU architecture are shown in Fig. {F:workflow} and Fig. {F:arch}. ', ''));
S2('Zero-shot cross-participant decoding'); P(C.R_ZS);
S2('Effect of few-shot calibration'); C.R_CAL.forEach(P);
S2('Ranking across participants'); P(C.R_RANK);
S2('Effect sizes'); P(C.R_EFFECT);
S2('Electrode reduction'); P(C.R_MONTAGE);
S2('Multi-metric evaluation'); P(C.R_MULTI);
S2('Variability across participants and seeds'); P(C.R_VAR); P(C.R_VARDEC);
S2('Who benefits from calibration'); P(C.R_CORR);
S2('Class-level behaviour and electrode importance'); P(C.R_CLASS2); P(C.R_SHAP);
S2('Representation geometry'); P(C.R_REPR);
S2('Secondary analyses'); P(C.R_SECONDARY);
S2('Comparison with published studies'); P(LIT_DISC);
S2('Implications'); P(C.DISCUSSION[1]); P(C.DISCUSSION[2]);
S2('Limitations'); P(C.DISCUSSION[3]);

S1('Conclusion and Future Work'); P(C.DISCUSSION[0] + ' ' + C.CONCLUSION);
P('Future work will address the limitations identified above:');
FUTURE.forEach(BUL);

const ALGO = {
  title: 'CLISA-EWU leave-one-participant-out training and evaluation',
  input: 'Standardised DE features *X* ∈ ℝ^N×d^, labels *y*, participant identifiers *s*, seeds {1, 7, 21}, calibration budget *m* = 20',
  output: 'Zero-shot accuracy A~zs~ and calibrated accuracy A~cal~ for every seed and held-out participant',
  steps: [
    ['for', 'each seed *r* ∈ (1, 7, 21)'],
    ['for', 'each participant *p* = 1, …, 15'],
    ['state', 'D~test~ ← windows of participant *p*; D~pool~ ← windows of the other 14 participants'],
    ['state', 'Split D~pool~ at random into D~train~ (85%) and D~val~ (15%)'],
    ['state', 'Initialise encoder *f*~θ~, projection head *g*~φ~ and classifier *c*~ψ~'],
    ['for', 'epoch = 1, …, 50', 'Stage 1: contrastive pretraining'],
    ['for', 'each class-balanced mini-batch *x* ⊂ D~train~'],
    ['state', '*v*~1~, *v*~2~ ← noise and electrode-masking views of *x*, Eq. {E:aug}'],
    ['state', 'Update θ, φ by minimising the NT-Xent loss of the projections of *v*~1~ and *v*~2~, Eq. {E:ntxent}'],
    ['endfor'],
    ['endfor'],
    ['for', 'epoch = 1, …, 50', 'Stage 2: supervised fine-tuning'],
    ['state', 'Update θ, ψ by minimising the cross-entropy loss on D~train~, Eq. {E:ce}'],
    ['if', 'validation macro F1 has not improved for 15 epochs'],
    ['state', 'stop training'],
    ['endif'],
    ['endfor'],
    ['state', 'Restore the best (θ, ψ); A~zs~(*r*, *p*) ← accuracy on D~test~', 'zero-shot'],
    ['state', 'Draw *m* windows D~cal~ ⊂ D~test~ at random; D~eval~ ← D~test~ ∖ D~cal~'],
    ['state', 'Fine-tune θ, ψ on D~cal~ for 10 epochs (Adam, learning rate 10^−3^)'],
    ['state', 'A~cal~(*r*, *p*) ← accuracy on D~eval~', 'calibrated'],
    ['endfor'],
    ['endfor'],
    ['return', 'A~zs~ and A~cal~ for all seeds and participants'],
  ],
};


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
const EQNUM = {}; B.filter((b) => b.t === 'eq').forEach((b, i) => { EQNUM[b.key] = i + 1; });
const render = (t) => t
  .replace(/Eq\. \{E:(\w+)\}/g, (_, k) => `Eq. (${EQNUM[k]})`)
  .replace(/Algorithm \{A:1\}/g, 'Algorithm 1')
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
const eqPara = (key) => { eqNo++; const [w, h] = EQSZ[key]; const sc = Math.min(0.30, 520 / w); return new Paragraph({ indent: { left: IND }, keepLines: true, spacing: { before: 120, after: 120 }, tabStops: [{ type: TabStopType.CENTER, position: Math.round(COL_W / 2) }, { type: TabStopType.RIGHT, position: COL_W }], children: [new TextRun({ text: '\t', font: SERIF }), new ImageRun({ type: 'png', data: fs.readFileSync(`${FIGDIR}/equations/eq_${key}.png`), transformation: { width: Math.round(w * sc), height: Math.round(h * sc) } }), new TextRun({ text: `\t(${eqNo})`, font: SERIF, size: BODY })] }); };

// ---------- Algorithm box ----------
function algoBlock() {
  const rule = (pos) => ({ [pos]: { style: BorderStyle.SINGLE, size: pos === 'top' ? 12 : 6, color: '000000', space: 2 } });
  const line = (children, o = {}) => { const tab = o.num ? 420 + (o.ind || 0) : 0; return new Paragraph({ indent: o.num ? { left: IND + tab, hanging: tab } : { left: IND }, spacing: { before: 0, after: 0, line: 252 }, keepNext: true, keepLines: true, border: o.border, tabStops: o.num ? [{ type: TabStopType.LEFT, position: tab }] : [], children }); };
  const outp = [line([new TextRun({ text: 'Algorithm 1 ', bold: true, font: SERIF, size: BODY }), ...runs(ALGO.title, { font: SERIF, size: BODY })], { border: { ...rule('top'), ...rule('bottom') } })];
  outp.push(line([new TextRun({ text: 'Input: ', bold: true, font: SERIF, size: 19 }), ...runs(render(ALGO.input), { font: SERIF, size: 19 })]));
  outp.push(line([new TextRun({ text: 'Output: ', bold: true, font: SERIF, size: 19 }), ...runs(render(ALGO.output), { font: SERIF, size: 19 })]));
  let depth = 0, n = 0;
  const kw = { for: ['for ', ' do'], if: ['if ', ' then'], state: ['', ''], return: ['return ', ''] };
  ALGO.steps.forEach(([t, text, cmt], k) => {
    if (t === 'endfor' || t === 'endif') { depth--; n++; outp.push(line([new TextRun({ text: `${n}:	`, font: SERIF, size: 19 }), new TextRun({ text: t === 'endfor' ? 'end for' : 'end if', bold: true, font: SERIF, size: 19 })], { ind: depth * 300, num: true })); return; }
    n++;
    const [pre, post] = kw[t];
    const ch = [new TextRun({ text: `${n}:	`, font: SERIF, size: 19 })];
    if (pre) ch.push(new TextRun({ text: pre, bold: true, font: SERIF, size: 19 }));
    ch.push(...runs(render(text), { font: SERIF, size: 19 }));
    if (post) ch.push(new TextRun({ text: post, bold: true, font: SERIF, size: 19 }));
    if (cmt) ch.push(new TextRun({ text: `   ▷ ${cmt}`, italics: true, font: SERIF, size: 19, color: '404040' }));
    outp.push(line(ch, { ind: depth * 300, num: true, border: k === ALGO.steps.length - 1 ? rule('bottom') : undefined }));
    if (t === 'for' || t === 'if') depth++;
  });
  outp.push(new Paragraph({ spacing: { after: 200 }, children: [] }));
  return outp;
}

// ---------- Assemble ----------
const out = [];
out.push(
  new Paragraph({ indent: { left: IND }, spacing: { before: 200, after: 240, line: 276, lineRule: d.LineRuleType.AUTO }, children: [new TextRun({ text: C.TITLE, bold: true, font: SANS, size: 46 })] }),
  new Paragraph({ indent: { left: IND }, spacing: { after: 200 }, children: runs(C.AUTHORS.replace(/\^1\*\^/, '^✉^').replace(/\^1\^/g, ''), { font: SANS, size: 21, bold: true }) }),
  new Paragraph({ indent: { left: IND }, spacing: { after: 60 }, children: [new TextRun({ text: 'Abstract', bold: true, font: SANS, size: 26 })] }),
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
  else if (b.t === 'bul') out.push(new Paragraph({ numbering: { reference: 'future', level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { before: 40, after: 40, line: 252 }, children: runs(render(b.text), { font: SERIF, size: BODY }) }));
  else if (b.t === 'algo') out.push(...algoBlock());
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
    { reference: 'future', levels: [{ level: 0, format: LevelFormat.BULLET, text: '\u2022', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: IND + 500, hanging: 300 } } } }] },
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
