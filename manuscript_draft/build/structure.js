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

// ---- Blocks ----
S1('Introduction');
C.INTRO.forEach((x) => P(x.replace(' (Table {T:lit})', '')));
P('This work addresses these gaps through a controlled benchmark with a rigorous, participant-level evaluation. The main contributions of this research are:');
CONTRIB_SR.forEach(([a, b]) => B.push({ t: 'num', text: `*${a}*: ${b}` }));
P(ORG2);

S1('Literature Review'); P(LIT_INTRO);
S2('Domain adaptation and participant invariance'); P(C.RELATED[0][1]);
S2('Graph and attention models'); P(C.RELATED[1][1]);
S2('Contrastive and self-supervised learning'); P(C.RELATED[2][1]);
S2('Reduced-electrode and wearable EEG'); P(C.RELATED[3][1]);
S2('Research gap'); P(GAP);

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
