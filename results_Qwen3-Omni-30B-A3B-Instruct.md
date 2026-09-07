200 scored, 0 unparsable/unk dropped

| Subset | n | Acc | Attr. correct | Attr. precision | Ghost evidence (reals) | Evidence / clip |
|---|---|---|---|---|---|---|
| ALL | 200 | 0.41 | 0.82 | 0.82 | 0.02 | 0.3 |
| FVRA | 50 | 0.16 | 0.25 | 0.30 | 0.00 | 0.2 |
| RVFA | 50 | 0.24 | 1.00 | 1.00 | 0.00 | 0.2 |
| FVFA | 50 | 0.26 | 1.00 | 1.00 | 0.00 | 0.3 |
| RVRA | 50 | 0.98 | 0.00 | 0.00 | 0.02 | 0.0 |

95% bootstrap CI (ALL):
  acc: 0.41 [0.34, 0.48]
  attr_correct: 0.82 [0.68, 0.94]
  attr_precision: 0.82 [0.69, 0.93]
  ghost_rate: 0.02 [0.00, 0.07]

Cited categories per quadrant (count; * = expected):
  FVRA: spatial_geometric*=2, temporal*=1, physiological*=0, cross_modal=7
  RVFA: spatial_geometric=0, temporal=0, physiological=0, cross_modal*=12
  FVFA: spatial_geometric*=2, temporal*=2, physiological=0, cross_modal*=13
  RVRA: spatial_geometric=0, temporal=0, physiological=0, cross_modal=1

Accuracy by #cited categories (true fakes):
  0 categories: n=117, recall=0.00
  1 categories: n=29, recall=1.00
  2 categories: n=2, recall=1.00
  3 categories: n=2, recall=1.00

% LaTeX
\begin{tabular}{lrrrrr}\toprule
Subset & $n$ & Acc & Attr.\ corr. & Attr.\ prec. & Ghost ev. \\\midrule
FVRA & 50 & 0.16 & 0.25 & 0.30 & -- \\
RVFA & 50 & 0.24 & 1.00 & 1.00 & -- \\
FVFA & 50 & 0.26 & 1.00 & 1.00 & -- \\
RVRA & 50 & 0.98 & -- & 0.00 & 0.02 \\
ALL & 200 & 0.41 & 0.82 & 0.82 & 0.02 \\
\bottomrule\end{tabular}
