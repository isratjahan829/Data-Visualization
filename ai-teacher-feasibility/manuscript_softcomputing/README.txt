Soft Computing (Springer) version of the manuscript -- Springer Nature LaTeX template (sn-jnl).

Overleaf (recommended):
  1. Overleaf -> New Project -> Templates -> search "Springer Nature LaTeX Template" -> Open as Template.
     (This gives you the publisher's sn-jnl.cls and sn-*.bst files.)
  2. In that project, delete the sample sn-article.tex and its sample bib/figures.
  3. Upload main.tex, refs.bib and the folders fig/ and generated/ from this zip
     (Upload -> drag the files; for folders create "fig" and "generated" first, then upload into them).
  4. Menu -> Main document: main.tex, Compiler: pdfLaTeX -> Recompile.

Reference style: author-year (option sn-basic), as used by Soft Computing. For a numbered style
change the class option to sn-mathphys-num.

Files: main.tex, refs.bib, fig/ (vector PDF figures), generated/ (numbers and tables produced by
the analysis code -- do not edit by hand).
Fill in before submission: [University name], [City], [Country], funding statement and the
generative-AI declaration (in "Declarations").
Soft Computing limits: abstract 150-250 words (this one is ~210); check current author guidelines
for page/figure limits and move e.g. the grid-verification, HTMT and invariance tables to Supplementary Information if needed.
