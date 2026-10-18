// Build an Overleaf-ready LaTeX version of the Scientific Reports draft.
// node tex.js <out_dir>   (writes main.tex; figures are copied separately)
const fs = require('fs');
const C = require('./content');
const OUT = process.argv[2];

// Reuse the section/table structure of the Word builder (srstyle.js) so both outputs stay identical.
const src = fs.readFileSync(__dirname + '/srstyle.js', 'utf8');
const slice = src.slice(src.indexOf('// ---------- Tables registry ----------'), src.indexOf('// ---------- Numbering by first mention ----------'));
const { B, TABLES, ALGO } = new Function('C', 'const TEXT_W = 1000;\n' + slice + '\nreturn { B, TABLES, ALGO };')(C);

// ---------- Markup → LaTeX ----------
const esc = (t) => t
  .replace(/\\/g, '\\textbackslash{}')
  .replace(/([&%$#_{}])/g, '\\$1')
  .replace(/R\u0304/g, '$\\bar{R}$')
  .replace(/\.\.\./g, '\\ldots{}');
function tex(text) {
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\[\[[^\]]+\]\]|\^[^^]+\^|~[^~]+~|(?:Fig\.|Figure) \{F:\w+\}|Eq\. \{E:\w+\}|Algorithm \{A:1\}|Table \{T:\w+\}|\{[a-z_,]+\})/g;
  let out = '', last = 0, m;
  while ((m = re.exec(text))) {
    out += esc(text.slice(last, m.index));
    const t = m[0];
    if (t.startsWith('**')) out += `\\textbf{${tex(t.slice(2, -2))}}`;
    else if (t.startsWith('[[')) out += `\\pending{${tex(t.slice(2, -2))}}`;
    else if (t.startsWith('*')) out += `\\emph{${tex(t.slice(1, -1))}}`;
    else if (t.startsWith('^')) out += `\\textsuperscript{${tex(t.slice(1, -1))}}`;
    else if (t.startsWith('~')) out += `\\textsubscript{${tex(t.slice(1, -1))}}`;
    else if (t.includes('{E:')) out += `Eq.~\\eqref{eq:${t.match(/\{E:(\w+)\}/)[1]}}`;
    else if (t.includes('{A:')) out += 'Algorithm~\\ref{alg:clisa}';
    else if (t.includes('{F:')) out += `Fig.~\\ref{fig:${t.match(/\{F:(\w+)\}/)[1]}}`;
    else if (t.includes('{T:')) out += `Table~\\ref{tab:${t.match(/\{T:(\w+)\}/)[1]}}`;
    else { const keys = t.slice(1, -1).split(','); keys.forEach((k) => { if (!C.REFS[k]) throw new Error('missing ref ' + k); if (!cited.includes(k)) cited.push(k); }); out += `\\cite{${keys.join(',')}}`; }
    last = m.index + t.length;
  }
  return out + esc(text.slice(last));
}
const cited = [];

// ---------- Equations ----------
const EQ = {
  de: String.raw`h(X)=-\int_{-\infty}^{\infty}f(x)\log f(x)\,\mathrm{d}x=\frac{1}{2}\log\left(2\pi e\,\sigma^{2}\right)`,
  zscore: String.raw`\tilde{x}_j=\frac{x_j-\mu_j}{\sigma_j},\qquad j=1,\ldots,d`,
  aug: String.raw`v_1=x+\varepsilon,\quad \varepsilon\sim\mathcal{N}(0,\,0.1^{2}I);\qquad v_2=x\odot(m\otimes\mathbf{1}_5),\quad m_c\sim\mathrm{Bernoulli}(0.7)`,
  encoder: String.raw`h=f_\theta(x)=\mathrm{ReLU}\Bigl(\mathrm{BN}\bigl(W_2\,\mathrm{Dropout}(\mathrm{ReLU}(\mathrm{BN}(W_1x)))\bigr)\Bigr),\qquad z=\frac{g_\phi(h)}{\lVert g_\phi(h)\rVert_2}`,
  ce: String.raw`\mathcal{L}_{\mathrm{CE}}=-\frac{1}{|\mathcal{B}|}\sum_{b\in\mathcal{B}}\sum_{i=1}^{C}y_{b,i}\log\hat{y}_{b,i}`,
  ntxent: String.raw`\ell_{i,j} = -\log \frac{\exp\left(\mathrm{sim}(z_i,z_j)/\tau\right)}{\sum_{k=1,\,k\neq i}^{2N}\exp\left(\mathrm{sim}(z_i,z_k)/\tau\right)}`,
  acc: String.raw`\mathrm{Accuracy}=\frac{1}{N}\sum_{i=1}^{C}TP_i,\qquad \mathrm{Specificity}_i=\frac{TN_i}{TN_i+FP_i}`,
  recall: String.raw`\mathrm{Recall}_i=\frac{TP_i}{TP_i+FN_i},\qquad \mathrm{Precision}_i=\frac{TP_i}{TP_i+FP_i}`,
  bacc: String.raw`\mathrm{BA}=\frac{1}{C}\sum_{i=1}^{C}\mathrm{Recall}_i`,
  f1: String.raw`F1_i=2\cdot\frac{\mathrm{Precision}_i\cdot\mathrm{Recall}_i}{\mathrm{Precision}_i+\mathrm{Recall}_i}`,
  macrof1: String.raw`F1_{\mathrm{macro}}=\frac{1}{C}\sum_{i=1}^{C}F1_i`,
  wf1: String.raw`F1_{\mathrm{weighted}}=\sum_{i=1}^{C}\frac{n_i}{N}\,F1_i`,
  kappa: String.raw`\kappa=\frac{p_o-p_e}{1-p_e},\qquad p_e=\frac{1}{N^{2}}\sum_{i=1}^{C}n_{i\cdot}\,n_{\cdot i}`,
  mcc: String.raw`\mathrm{MCC}=\frac{c\,N-\sum_{i}p_i\,t_i}{\sqrt{\left(N^{2}-\sum_{i}p_i^{2}\right)\left(N^{2}-\sum_{i}t_i^{2}\right)}}`,
  gmean: String.raw`G=\Bigl(\prod_{i=1}^{C}\mathrm{Recall}_i\Bigr)^{1/C}`,
  itr: String.raw`B=\log_2 C+P\log_2 P+(1-P)\log_2\frac{1-P}{C-1},\qquad \mathrm{ITR}=B\cdot\frac{60}{T}`,
  retention: String.raw`\mathrm{Retention}=\frac{\mathrm{Acc}_{6}}{\mathrm{Acc}_{62}}`,
  friedman: String.raw`\chi^{2}_{F}=\frac{12n}{k(k+1)}\Biggl[\sum_{j=1}^{k}\bar{R}_j^{2}-\frac{k(k+1)^{2}}{4}\Biggr],\qquad W=\frac{\chi^{2}_{F}}{n(k-1)}`,
  cd: String.raw`\mathrm{CD}=q_{\alpha}\sqrt{\frac{k(k+1)}{6n}}`,
  effect: String.raw`d_z=\frac{\bar{D}}{s_D},\qquad r_{rb}=\frac{T^{+}-T^{-}}{T^{+}+T^{-}}`,
  icc: String.raw`\mathrm{ICC}(3,1)=\frac{MS_P-MS_E}{MS_P+(k-1)\,MS_E}`,
};

// ---------- Figures and tables ----------
const figure = (k) => {
  const f = C.FIGS[k]; const tall = f.h / f.w > 0.9;
  return `\\begin{figure}[!htbp]
  \\centering
  \\includegraphics[width=${tall ? '0.82' : '1'}\\linewidth,height=0.62\\textheight,keepaspectratio]{figures/${f.file}}
  \\caption{\\textbf{${tex(f.title)}} ${tex(f.text)}}
  \\label{fig:${k}}
\\end{figure}\n`;
};
const table = (k) => {
  const t = TABLES[k]; const n = t.header.length;
  const size = n >= 11 ? '\\scriptsize' : n >= 8 ? '\\footnotesize' : '\\small';
  const sep = n >= 11 ? '2.6pt' : n >= 8 ? '3.5pt' : '5pt';
  const wide = n <= 4;
  const spec = k === 'families' ? '@{}>{\\raggedright\\arraybackslash}X>{\\raggedright\\arraybackslash}X>{\\raggedright\\arraybackslash}X>{\\raggedright\\arraybackslash}X@{}' : k === 'lit' ? '@{}llXl@{}' : wide ? `@{}l${'X'.repeat(n - 1)}@{}` : `@{}l${'c'.repeat(n - 1)}@{}`;
  const env = wide ? ['\\begin{tabularx}{\\linewidth}{' + spec + '}', '\\end{tabularx}'] : ['\\begin{tabular}{' + spec + '}', '\\end{tabular}'];
  let body = t.header.map((h) => `\\textbf{${tex(h)}}`).join(' & ') + ' \\\\\n\\midrule\n';
  t.rows.forEach((r, i) => {
    if (t.group && r[1] === '') body += (i ? '\\midrule\n' : '') + `\\multicolumn{${n}}{@{}l}{\\textit{${tex(r[0])}}} \\\\\n`;
    else body += r.map((c) => tex(String(c))).join(' & ') + ' \\\\\n';
  });
  return `\\begin{table}[!htbp]
  \\centering
  {${size}\\setlength{\\tabcolsep}{${sep}}
  ${env[0]}
  \\toprule
${body}  \\bottomrule
  ${env[1]}}
  \\caption{${tex(t.title)}}
  \\label{tab:${k}}
\\end{table}\n`;
};

// ---------- Algorithm ----------
function algo() {
  let o = `\\begin{algorithm}[!htbp]\n\\caption{${tex(ALGO.title)}}\n\\label{alg:clisa}\n\\small\n\\begin{algorithmic}[1]\n\\Require ${tex(ALGO.input)}\n\\Ensure ${tex(ALGO.output)}\n`;
  for (const [t, text, cmt] of ALGO.steps) {
    const c = cmt ? ` \\Comment{${tex(cmt)}}` : '';
    if (t === 'for') o += `\\For{${tex(text)}}${c}\n`;
    else if (t === 'if') o += `\\If{${tex(text)}}${c}\n`;
    else if (t === 'endfor') o += '\\EndFor\n';
    else if (t === 'endif') o += '\\EndIf\n';
    else if (t === 'return') o += `\\State \\Return ${tex(text)}\n`;
    else o += `\\State ${tex(text)}${c}\n`;
  }
  return o + '\\end{algorithmic}\n\\end{algorithm}\n\n';
}

// ---------- Body ----------
let body = '', inList = false, inBul = false;
const strip = (x) => x.replace(/^\d+(\.\d+)*\s+/, '');
const placedF = new Set(), placedT = new Set();
for (const b of B) {
  if (b.t !== 'num' && inList) { body += '\\end{enumerate}\n\n'; inList = false; }
  if (b.t !== 'bul' && inBul) { body += '\\end{itemize}\n\n'; inBul = false; }
  if (b.t === 'h1') body += `\n\\section{${tex(strip(b.text))}}\n`;
  else if (b.t === 'h2') body += `\n\\subsection{${tex(strip(b.text))}}\n`;
  else if (b.t === 'bul') { if (!inBul) { body += '\\begin{itemize}\n'; inBul = true; } body += `  \\item ${tex(b.text)}\n`; }
  else if (b.t === 'algo') body += algo();
  else if (b.t === 'h3') body += `\n\\paragraph{${tex(b.text)}}\n`;
  else if (b.t === 'eq') body += `\\begin{equation}\n  ${EQ[b.key]}\n  \\label{eq:${b.key}}\n\\end{equation}\n`;
  else if (b.t === 'num') { if (!inList) { body += '\\begin{enumerate}\n'; inList = true; } body += `  \\item ${tex(b.text)}\n`; }
  else body += tex(b.text) + '\n\n';
  if (b.text) {
    for (const m of b.text.matchAll(/\{T:(\w+)\}/g)) if (!placedT.has(m[1])) { placedT.add(m[1]); body += table(m[1]) + '\n'; }
    for (const m of b.text.matchAll(/\{F:(\w+)\}/g)) if (!placedF.has(m[1])) { placedF.add(m[1]); body += figure(m[1]) + '\n'; }
  }
}
if (inList) body += '\\end{enumerate}\n\n';
if (inBul) body += '\\end{itemize}\n\n';
// Captions are rendered when their floats are placed; make sure citations inside them are registered in order.
const back = String.raw`
\section*{Data availability}
The SEED-IV dataset is available from its providers under their licence at \url{https://bcmi.sjtu.edu.cn/home/seed/}. Data splits, per-fold predictions, confusion matrices and result tables are available at \url{https://github.com/saifkhancse/clisa-ewu-brain-informatics} \pending{archive a tagged release on Zenodo and cite its DOI}.

\section*{Code availability}
Model, training, evaluation and analysis code, including the scripts that regenerate every statistic and figure in this paper, is available at \url{https://github.com/saifkhancse/clisa-ewu-brain-informatics}.
`;
const bib = '\\begin{thebibliography}{99}\n' + cited.map((k) => `\\bibitem{${k}} ${tex(C.REFS[k])}`).join('\n\n') + '\n\\end{thebibliography}\n';
const end = String.raw`
\section*{Acknowledgements}
The authors thank the providers of the SEED-IV dataset and the Department of Computer Science and Engineering, East West University, for their support. All figures were created by the authors; schematic illustrations (Figs.~\ref{fig:workflow} and \ref{fig:arch}) were drawn manually and all plots were generated with Python.

\section*{Author contributions}
S.K.: conceptualization, methodology, software, experiments, formal analysis, visualization and writing of the original draft. O.B.Q.: experiments, analysis and manuscript preparation. K.M.F.A.B.: implementation, evaluation and documentation. R.R.: experiments, reporting and software integration (NeuroSync). A.W.R.: supervision, review and editing. All authors reviewed and approved the final manuscript.

\section*{Funding}
No external funding was received for this study \pending{confirm}.

\section*{Declarations}
\subsection*{Competing interests}
The authors declare no competing interests.

\subsection*{Additional information}
\textbf{Correspondence} and requests for materials should be addressed to S.K.
`;

const preamble = String.raw`% =====================================================================
% Scientific Reports manuscript draft (Overleaf-ready)
% Compile with pdfLaTeX (Overleaf default). Figures live in ./figures.
% Red bracketed text (\pending{...}) marks results still pending or facts to verify.
%
% To move into the official Springer Nature template (sn-jnl.cls, available
% in the Overleaf gallery as "Springer Nature LaTeX Template"): copy the
% \title/\author block, the abstract, everything between \begin{document}
% and \end{document}, and the thebibliography block; select the
% "sn-nature" reference style option in that template.
% =====================================================================
\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{newunicodechar}
\usepackage{mathptmx}
\usepackage[scaled=0.92]{helvet}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs,tabularx,array}
\usepackage{xcolor}
\usepackage[super,sort&compress,comma]{natbib}
\usepackage{authblk}
\usepackage{caption}
\usepackage{lineno}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{url}

% Scientific Reports style: unnumbered sans-serif headings, captions "Fig. 1." / "Table 1."
\setcounter{secnumdepth}{2}
\titleformat{\section}{\sffamily\bfseries\large}{\thesection}{0.6em}{}
\titleformat{\subsection}{\sffamily\bfseries\normalsize}{\thesubsection}{0.6em}{}
\titleformat{\paragraph}[hang]{\itshape}{}{0pt}{}
\titlespacing*{\paragraph}{0pt}{6pt}{2pt}
\captionsetup{font=small,labelfont=bf,labelsep=period,justification=justified,singlelinecheck=false}
\captionsetup[figure]{name=Fig.}
\setlist[enumerate]{leftmargin=*,itemsep=2pt}
\setlength{\parindent}{1.2em}
\renewcommand{\arraystretch}{1.12}
\newcommand{\pending}[1]{\textcolor{red}{[#1]}}
\setlength{\emergencystretch}{1.5em}

% Unicode characters used in the text
\newunicodechar{κ}{\ensuremath{\kappa}}
\newunicodechar{ρ}{\ensuremath{\rho}}
\newunicodechar{τ}{\ensuremath{\tau}}
\newunicodechar{χ}{\ensuremath{\chi}}
\newunicodechar{α}{\ensuremath{\alpha}}
\newunicodechar{Δ}{\ensuremath{\Delta}}
\newunicodechar{ℓ}{\ensuremath{\ell}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{√}{\ensuremath{\surd}}
\newunicodechar{≠}{\ensuremath{\neq}}
\newunicodechar{θ}{\ensuremath{\theta}}
\newunicodechar{φ}{\ensuremath{\phi}}
\newunicodechar{ψ}{\ensuremath{\psi}}
\newunicodechar{σ}{\ensuremath{\sigma}}
\newunicodechar{μ}{\ensuremath{\mu}}
\newunicodechar{ε}{\ensuremath{\varepsilon}}
\newunicodechar{ℝ}{\ensuremath{\mathbb{R}}}
\newunicodechar{∈}{\ensuremath{\in}}
\newunicodechar{⊂}{\ensuremath{\subset}}
\newunicodechar{∖}{\ensuremath{\setminus}}

\title{\sffamily\bfseries ${tex(C.TITLE)}}
\author[1,*]{Saif Khan}
\author[1]{Osama Bin Qashem}
\author[1]{K. M. Fahim A. Bari}
\author[1]{Rafid Rahman}
\author[1]{Ahmed Wasif Reza}
\affil[1]{Department of Computer Science and Engineering, East West University, Dhaka 1212, Bangladesh}
\affil[*]{Corresponding author: \href{mailto:saifkhantdm@gmail.com}{saifkhantdm@gmail.com}}
\date{}

\begin{document}
\maketitle
\linenumbers

\begin{abstract}
\noindent\textbf{${tex(C.ABSTRACT_SR)}}
\end{abstract}

\noindent\textsf{\textbf{Keywords}}\quad ${tex(C.KEYWORDS.map((k, i) => (i ? k : k[0].toUpperCase() + k.slice(1))).join(', '))}

\medskip
`;

let doc = preamble + body + back + '\n' + bib + end + '\n\\end{document}\n';
// Author edits carried over from the Overleaf version: placeholders the authors resolved or chose to drop.
const DROP = ['report sex and age range', 'confirm band limits', 'confirm that μ and σ', 'either implement cross-participant', 'pending: trial-held-out', 'pending ablation: calibrate', 'version', 'confirm hardware', 'recompute on held-out', 'Re-run before submission'];
doc = doc.replace(/ ?\\pending\{((?:[^{}]|\{[^{}]*\})*)\}/g, (m, inner) => (DROP.some((x) => inner.includes(x)) ? '' : m));
doc = doc.replace('All figures were created by the authors; schematic illustrations (Figs.~\\ref{fig:workflow} and \\ref{fig:arch}) were drawn manually and all plots were generated with Python.',
  'The authors created all figures; they drew the schematic illustrations (Figs.~\\ref{fig:workflow} and \\ref{fig:arch}) manually, and they generated all plots in Python.');
fs.writeFileSync(`${OUT}/main.tex`, doc);
const figs = [...placedF].map((k) => C.FIGS[k].file);
fs.writeFileSync(`${OUT}/figure_list.txt`, figs.join('\n') + '\n');
console.log('main.tex written:', doc.length, 'chars | figures', figs.length, '| tables', placedT.size, '| refs', cited.length, '| equations', B.filter((b) => b.t === 'eq').length);
