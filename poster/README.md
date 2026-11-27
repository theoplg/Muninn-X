# A1 poster (portrait, 59.4 x 84.1 cm)

- `poster.tex`: beamer + beamerposter, self-contained (no external theme). Title band, abstract, four
  asymmetric quadrants (1 Muninn | 3 proposed approach / 2 Muninn on flow matching | 4 Maze2D results), footer
  with references. Body text ~19 pt (= ~26 pt on A0); Times-like text and math (newtx), Helvetica-like headings.
- `figures/*.pdf`: vector figures drawn at their printed size (STIX fonts, same method colours everywhere),
  regenerated from the results in `runs/` by `python scripts/poster_figures.py` (the maze and amplification
  panels need the GPU). Diagrams (Muninn chain, reuse vs extrapolation) and algorithms are TikZ / LaTeX.
- Build: `latexmk -pdf poster.tex`. Overleaf: upload `poster.tex` + `figures/`, compiler pdfLaTeX.

To fill in: authors, affiliations, e-mail, logos (dashed boxes in the title band), acknowledgements, code link.
Still to produce: the dashed "Figure to produce" box in section 4 (plans of each certified method on the same
contexts; needs a small sampling script and the E19 regressor saved or retrained).
Readability check: print the PDF on A4; if it reads at arm's length, the A1 reads from ~1-1.5 m.
