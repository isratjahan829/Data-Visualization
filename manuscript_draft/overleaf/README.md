# Overleaf project: Scientific Reports draft

1. In Overleaf choose **New Project → Upload Project** and upload `CLISA-EWU_SciRep_overleaf.zip`.
2. Compiler: **pdfLaTeX** (Overleaf default). Main file: `main.tex`.
3. Red bracketed text (`\pending{...}`) marks results still pending or facts to verify.

Contents: `main.tex` (manuscript, 21 equations, Algorithm 1, 10 tables, embedded bibliography) and `figures/` (21 PNG figures).
To switch to the official Springer Nature template (`sn-jnl.cls`, Overleaf gallery), see the comment block at the top of `main.tex`.
`main.tex` is generated from `../build/content.js` by `../build/tex.js`; regenerate rather than hand-editing if the shared content changes.
