# LEAF initial paper draft

Open `main.tex` as the main document. Compile with pdfLaTeX and BibTeX; in Overleaf, upload the ZIP and select `main.tex`.

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

This version turns the narrative outline into an English IEEE journal working draft. It includes the three selected image-generated conceptual illustrations and the audited main results table. Two clearly marked figure slots reserve the planned experimental displays. The slots contain descriptions, with no fabricated chart data.

Contents:

- `sections/`: editable manuscript sections.
- `figures/`: the three selected raster illustrations.
- `data/`: existing experiment summaries for the next plotting iteration.
- `references.bib`: the established bibliography.
- `LEAF_Draft_Outline.md`: paragraph purposes and manuscript structure.
- `LEAF_Draft_Discussion.md`: author questions and evidence needs.
- `main.pdf`: compiled preview.

The main document retains the method and protocol in the body. It has no supplementary-material dependency. The draft table uses arithmetic means and sample standard deviations across the actual trained seeds. Captions distinguish illustrative icon counts from actual evaluation configurations.

This initial draft describes the counts-on, query-history stop-gradient version. The current LEAF manuscript uses the saved NoMem final models; its formal results and historical method identities are documented in the [main experiment report](../../docs/LEAF_主实验结果分析.md). Align any revision of this draft with the intended checkpoint version and the baseline definitions in `LEAF_Draft_Discussion.md`.

The current experiment archive is `Open-SCORE/outputs/main/`: HAD episode records are in [`had/episodes.csv`](../../Open-SCORE/outputs/main/had/episodes.csv), and model identities, archived paths, and source provenance are in [`summary.json`](../../Open-SCORE/outputs/main/summary.json). The initial draft's manuscript and data files remain its historical publication snapshot.
