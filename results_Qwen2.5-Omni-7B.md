200 scored, 0 unparsable/unk dropped

| Subset | n | Acc | Attr. correct | Attr. precision | Ghost evidence (reals) | Evidence / clip |
|---|---|---|---|---|---|---|
| ALL | 200 | 0.27 | 1.00 | 0.80 | 0.00 | 0.0 |
| FVRA | 50 | 0.02 | 1.00 | 0.50 | 0.00 | 0.0 |
| RVFA | 50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.0 |
| FVFA | 50 | 0.04 | 1.00 | 1.00 | 0.00 | 0.1 |
| RVRA | 50 | 1.00 | 0.00 | 0.00 | 0.00 | 0.0 |

95% bootstrap CI (ALL):
  acc: 0.27 [0.20, 0.33]
  attr_correct: 1.00 [0.00, 1.00]
  attr_precision: 0.80 [0.00, 1.00]
  ghost_rate: 0.00 [0.00, 0.00]

Cited categories per quadrant (count; * = expected):
  FVRA: spatial_geometric*=0, temporal*=1, physiological*=0, cross_modal=1
  RVFA: spatial_geometric=0, temporal=0, physiological=0, cross_modal*=0
  FVFA: spatial_geometric*=0, temporal*=1, physiological=0, cross_modal*=2
  RVRA: spatial_geometric=0, temporal=0, physiological=0, cross_modal=0

Accuracy by #cited categories (true fakes):
  0 categories: n=147, recall=0.00
  1 categories: n=1, recall=1.00
  2 categories: n=2, recall=1.00

% LaTeX
\begin{tabular}{lrrrrr}\toprule
Subset & $n$ & Acc & Attr.\ corr. & Attr.\ prec. & Ghost ev. \\\midrule
FVRA & 50 & 0.02 & 1.00 & 0.50 & -- \\
RVFA & 50 & 0.00 & 0.00 & 0.00 & -- \\
FVFA & 50 & 0.04 & 1.00 & 1.00 & -- \\
RVRA & 50 & 1.00 & -- & 0.00 & 0.00 \\
ALL & 200 & 0.27 & 1.00 & 0.80 & 0.00 \\
\bottomrule\end{tabular}
