# A1 poster (landscape, 84.1 x 59.4 cm)

- `poster.tex`: beamer + beamerposter + tcolorbox, four asymmetric quadrants (Muninn | FM experiments | proposition | results), Lato font. Dashed boxes = figures still to produce.
- `figures/*.pdf`: vector figures, regenerated from the results in `runs/` by
  `python scripts/poster_figures.py` (the maze and amplification panels need the GPU).
- Build locally: `latexmk -pdf poster.tex`. On Overleaf: upload `poster.tex` + `figures/`, compiler pdfLaTeX.

Placeholders to fill: authors, affiliation, contact, full reference for [1] (Muninn).
Readability check: print the PDF on A4; if it reads at arm's length, the A1 reads from ~2 m.
