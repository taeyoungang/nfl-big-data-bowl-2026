# Expected YAC, completion probability, pass rush and receiver opportunity

Receiver rows: 31,539 on 6,805 thrown passes in 122 games (targets: 6,804; completions with a YAC label: 4,396). All scores are out-of-fold: 5 folds grouped by game.
Two model versions. **Analysis** also sees the first 0.5 s of ball flight. **Decision** sees only what was knowable at release and is defined identically for every eligible receiver, so it is the one used to score receivers who were not thrown to.

## 1. Expected YAC (point estimate)
Target: yards after catch, capped at 30 (0.7% of catches exceed it). Evaluated on 4,396 held-out catches.
```
                                      model    n   MAE  RMSE     R2  Spearman
                             A. Average YAC 4396 3.905 5.399 -0.001    -0.046
        B. nflfastR xYAC (public benchmark) 4228 3.745 5.190  0.087     0.324
                    A on the same rows as B 4228 3.890 5.438 -0.002    -0.052
                 1. Context + receiver role 4396 3.726 5.290  0.039     0.246
                            2. + Separation 4396 3.329 4.944  0.160     0.453
        3. + Open field, leverage, blockers 4396 3.164 4.771  0.218     0.516
          4. + Receiver motion and location 4396 2.939 4.528  0.296     0.618
           5. + Projected arrival and throw 4396 2.821 4.408  0.333     0.653
6. + Pass rush and timing  [analysis model] 4396 2.815 4.407  0.333     0.656
     7. Release-time only  [decision model] 4396 2.980 4.576  0.281     0.608
```
Earlier probe, which used nflverse air_yards as an input: MAE 2.93, R2 0.29. These models use no nflverse input except the label.

## 2. Expected YAC as a range
**Analysis model**
```
          share of catches below  model pinball  unconditional pinball  improvement
quantile                                                                           
0.10                       0.207          0.454                  0.451       -0.006
0.25                       0.288          1.019                  1.109        0.081
0.50                       0.540          1.364                  1.878        0.274
0.75                       0.751          1.398                  1.916        0.271
0.90                       0.886          1.040                  1.343        0.225
```
10th-90th band holds 87.8% of catches (nominal 80%); 25th-75th band holds 63.4% (nominal 50%). 27.3% of catches gain 0 yd or less, so the low quantiles sit at about 0 and the bands over-cover; the upper quantiles are the informative ones (share of catches below the 75th and 90th percentile predictions: 75.1% and 88.6%).
P(YAC >= 10): base rate 13.1% | AUC 0.830 | Brier 0.0902 vs 0.1142 for the base rate | top decile: 50.6% gain 10 or more.

**Decision model**
```
          share of catches below  model pinball  unconditional pinball  improvement
quantile                                                                           
0.10                       0.208          0.453                  0.451       -0.005
0.25                       0.297          1.002                  1.109        0.096
0.50                       0.529          1.439                  1.878        0.234
0.75                       0.747          1.467                  1.916        0.234
0.90                       0.884          1.086                  1.343        0.192
```
10th-90th band holds 87.8% of catches (nominal 80%); 25th-75th band holds 57.9% (nominal 50%). 27.3% of catches gain 0 yd or less, so the low quantiles sit at about 0 and the bands over-cover; the upper quantiles are the informative ones (share of catches below the 75th and 90th percentile predictions: 74.7% and 88.4%).
P(YAC >= 10): base rate 13.1% | AUC 0.809 | Brier 0.0950 vs 0.1142 for the base rate | top decile: 46.0% gain 10 or more.

Calibration of P(YAC >= 10), analysis model:
```
       n  predicted  actual
bin                        
0    440      0.005   0.000
1    440      0.015   0.014
2    439      0.025   0.027
3    440      0.037   0.043
4    439      0.054   0.071
5    440      0.078   0.084
6    439      0.113   0.098
7    440      0.169   0.186
8    439      0.263   0.287
9    440      0.515   0.505
```

## 3. Completion probability and pass rush
6,804 targeted attempts (complete, incomplete, interception); completion rate 64.9%.
```
                                                       AUC  log loss   Brier
model                                                                       
Base rate                                           0.5000    0.6480  0.2277
C1. Context, role, throw distance                   0.6525    0.6159  0.2134
C2. + receiver geometry at release                  0.7620    0.5427  0.1803
C3. + pass rush and timing  [decision model]        0.7794    0.5291  0.1751
C4. + first 0.5 s of ball flight  [analysis model]  0.8026    0.5036  0.1643
C5. + PFF pressure flag (charged after the play)    0.8047    0.5021  0.1637
```
Pass-rush features are measured at release: distance from the quarterback to the nearest rusher, rushers within 5 yd, the nearest rusher's closing speed, number of rushers, quarterback depth and speed, and time from snap to throw. C2 to C3 isolates their contribution. The PFF flag is charged after the play and is shown only to size what it adds beyond the geometry.
Calibration of the decision-model completion probability:
```
       n  predicted  actual
bin                        
0    681      0.205   0.222
1    680      0.344   0.332
2    680      0.470   0.453
3    681      0.596   0.577
4    680      0.691   0.718
5    680      0.755   0.765
6    681      0.805   0.808
7    680      0.842   0.853
8    680      0.876   0.860
9    681      0.916   0.905
```
**Nearest rusher to the QB at release**
```
          attempts  completion_rate  model_p_comp  mean_yac
rush_bin                                                   
< 2 yd        1282            0.516         0.533     4.352
2-4           4307            0.662         0.660     4.258
4-6           1024            0.745         0.742     4.659
6-8            130            0.746         0.712     5.351
8+              61            0.738         0.733     6.182
```
**Number of pass rushers**
```
            attempts  completion_rate  model_p_comp  mean_yac
nrush_bin                                                    
3 or fewer       357            0.664         0.665     4.143
4               4795            0.661         0.662     4.223
5               1317            0.625         0.625     5.034
6+               334            0.557         0.566     4.597
```
**Time from snap to throw**
```
         attempts  completion_rate  model_p_comp  mean_yac
ttt_bin                                                   
< 2.0 s       808            0.772         0.760     3.889
2.0-2.5      1687            0.688         0.687     3.981
2.5-3.0      1804            0.630         0.631     4.636
3.0-3.5      1191            0.631         0.631     4.827
3.5-4.0       607            0.598         0.608     4.648
4.0+          688            0.533         0.549     4.605
```
**By the PFF pressure flag (charged after the play)**
```
           attempts  completion_rate  mean_yac  yac_over_expected
pressured                                                        
0              4843            0.698     4.390              0.051
1              1961            0.529     4.368             -0.198
```

## 4. Which feature families matter
```
                              YAC (analysis model): MAE added when scrambled, yd  Completion (decision model): log loss added when scrambled
Flight-based projection                                                   0.9647                                                      0.0000
Receiver motion and location                                              0.2052                                                      0.0132
Decision projection                                                       0.0717                                                      0.0950
Open field                                                                0.0476                                                      0.0100
Receiver role and body                                                    0.0244                                                      0.0010
Separation                                                                0.0228                                                      0.0024
Leverage                                                                  0.0136                                                      0.0006
Pass rush and timing                                                      0.0111                                                      0.0346
Game context                                                              0.0085                                                      0.0001
Throw                                                                     0.0081                                                      0.0025
Blockers ahead                                                            0.0023                                                     -0.0000
```
Scrambling a whole family overstates its unique value when families overlap. The projection features contain the same positions and velocities as the t0 and t5 families, so the unique gain is the one in the section 1 ladder (step 4 to step 5: 2.94 to 2.82 yd of error), not the scrambled-family figure.

## 5. Coverage scheme: what happens relative to what the geometry predicts
YAC over expected = actual YAC minus the analysis model's expectation, by PFF coverage. Interval = 95%, bootstrap by game. Family-wise p comes from shuffling coverage labels within each game 2,000 times and asking how often the largest |z| across all 7 schemes is as big (a |z| above 2.68 is significant at 5% after allowing for 7 comparisons). Overall mean residual -0.008.
```
             n   mean     lo     hi  raw mean YAC      z  family-wise p
Cover-0    119 -0.484 -1.145  0.191         3.353 -1.208          0.816
Cover-1    886  0.619  0.272  0.961         5.097  4.212          0.000
Cover-2    619 -0.866 -1.155 -0.579         4.273 -4.929          0.000
Cover-3   1499 -0.067 -0.265  0.136         4.468 -0.591          0.995
Cover-6    451  0.160 -0.215  0.558         4.390  0.777          0.980
Quarters   541  0.007 -0.363  0.408         3.926  0.036          1.000
Red Zone   152 -0.414 -0.788 -0.061         1.697 -1.168          0.848
```

Cover-2 minus other zone coverages (Cover-3, Cover-6, Quarters): -0.86 yd (95% interval -1.18 to -0.54). Cover-1 minus other man coverages (Cover-0, 2-Man): +0.61 yd (95% interval -0.15 to +1.35).

**Cover-2 subgroups** (n = 619): mean YAC over expected
```
                                 split    n  mean    lo    hi
3-8 yd                Depth at arrival  225 -0.70 -1.13 -0.19
8+ yd                 Depth at arrival  184 -0.93 -1.45 -0.29
behind LOS to 3 yd    Depth at arrival  210 -0.99 -1.42 -0.52
1st                               Down  246 -0.87 -1.33 -0.31
2nd                               Down  237 -0.81 -1.25 -0.36
3rd/4th                           Down  136 -0.96 -1.54 -0.30
RB                            Receiver  187 -1.04 -1.49 -0.57
TE                            Receiver  161 -1.00 -1.42 -0.55
WR                            Receiver  270 -0.69 -1.11 -0.20
Shotgun/empty/pistol         Formation  515 -0.86 -1.15 -0.53
Under center                 Formation  104 -0.92 -1.60 -0.16
3-4 rushers                    Rushers  575 -0.95 -1.22 -0.65
5+ rushers                     Rushers   44  0.23 -1.07  1.89
2.7 s or more            Time to throw  353 -1.00 -1.41 -0.61
under 2.7 s              Time to throw  266 -0.69 -1.03 -0.33
```
**Cover-1 subgroups** (n = 886): mean YAC over expected
```
                                 split    n  mean    lo    hi
3-8 yd                Depth at arrival  284  0.77  0.20  1.40
8+ yd                 Depth at arrival  416  0.08 -0.43  0.67
behind LOS to 3 yd    Depth at arrival  186  1.59  0.71  2.47
1st                               Down  241  1.23  0.62  1.84
2nd                               Down  254  0.39 -0.12  0.95
3rd/4th                           Down  391  0.39 -0.21  0.96
RB                            Receiver  123  2.18  0.95  3.37
TE                            Receiver  173  0.17 -0.41  0.75
WR                            Receiver  588  0.41  0.01  0.85
Shotgun/empty/pistol         Formation  748  0.51  0.14  0.87
Under center                 Formation  138  1.22  0.44  2.03
3-4 rushers                    Rushers  524  0.46  0.01  0.93
5+ rushers                     Rushers  362  0.85  0.27  1.45
2.7 s or more            Time to throw  384  0.72  0.17  1.33
under 2.7 s              Time to throw  502  0.54  0.16  0.94
```
Does the scheme effect depend on the baseline? Mean YAC minus expected YAC, three different expectations:
```
          analysis model  decision model (release only)  nflfastR xYAC (no tracking)
Cover-0            -0.48                          -0.55                          NaN
Cover-1             0.62                           0.67                         0.36
Cover-2            -0.87                          -0.92                        -1.03
Cover-3            -0.07                          -0.05                        -0.56
Cover-6             0.16                           0.16                        -0.58
Quarters            0.01                           0.06                        -0.80
Red Zone           -0.41                          -0.40                          NaN
```
The nflfastR column is offset downward for every scheme (its expectation is fit to all plays, including the screens that are missing here), so compare each scheme with the others in that column and not with zero. Cover-2 is the clear low outlier on all three baselines.
95% intervals for the two flagged schemes: analysis model: Cover-2 -0.87 (-1.15 to -0.58), Cover-1 +0.62 (+0.27 to +0.96); decision model (release only): Cover-2 -0.92 (-1.24 to -0.59), Cover-1 +0.67 (+0.32 to +1.04); nflfastR xYAC (no tracking): Cover-2 -1.03 (-1.37 to -0.69), Cover-1 +0.36 (-0.06 to +0.76)

Is Cover-2 simply a different situation? Average geometry on Cover-2 vs other zone catches:
```
grp            Cover-2  Other zone  difference
sep1_p            5.30        4.41        0.89
n_def10_p         2.19        2.29       -0.10
t_close_p         1.53        1.28        0.25
cone_dist_p       9.79        9.66        0.13
n_block10_p       0.13        0.14       -0.01
lev_dx_p          2.01        1.68        0.33
x_rel_p           6.72        7.51       -0.79
tau               0.90        0.94       -0.04
time_to_throw     2.88        2.80        0.08
n_rush            3.96        4.16       -0.20
```
Average geometry on Cover-1 vs other man catches:
```
grp            Cover-1  Other man  difference
sep1_p            3.17       3.59       -0.42
n_def10_p         2.14       1.96        0.18
t_close_p         0.99       1.13       -0.15
cone_dist_p      12.14      14.70       -2.56
n_block10_p       0.16       0.17       -0.01
lev_dx_p          1.18       1.01        0.17
x_rel_p           8.83       9.50       -0.67
tau               1.03       1.11       -0.09
time_to_throw     2.73       2.59        0.14
n_rush            4.44       4.92       -0.48
```

Completion rate over expected by coverage (actual minus the analysis model's probability):
```
             n   mean     lo     hi  completion rate  model p_comp
2-Man      139  0.015 -0.050  0.079            0.612         0.596
Cover-0    225 -0.002 -0.059  0.052            0.556         0.558
Cover-1   1571  0.013 -0.011  0.033            0.568         0.556
Cover-2    891 -0.041 -0.065 -0.015            0.695         0.735
Cover-3   2156  0.011 -0.007  0.028            0.696         0.685
Cover-6    656  0.002 -0.025  0.032            0.688         0.686
Quarters   815 -0.008 -0.039  0.023            0.664         0.672
Red Zone   275 -0.030 -0.083  0.028            0.578         0.608
```

## 6. Receiver opportunity: scoring every eligible receiver at release (decision model)
Sanity check. Targets should look more open than the receivers who were not thrown to, using only release-time measurements:
```
             target  not targeted
p_comp_d      0.650         0.551
yac_mean_d    3.941         3.790
sep1_t0       3.791         3.175
sep1_d        4.207         3.601
t_close_d     1.344         1.173
cone_dist_d  14.038        14.621
```
```
                                                               target ranked first  random pick  mean rank of target  mean eligible
receiver ranking                                                                                                                   
highest completion probability                                               0.360        0.222                2.309          4.636
highest YAC opportunity (p x E[YAC])                                         0.292        0.222                2.577          4.636
highest expected yards, uncapped depth (p x (depth + E[YAC]))                0.277        0.222                2.565          4.636
highest expected yards, depth credit capped at 20 yd                         0.284        0.222                2.539          4.636
```
Gap between the best option and the receiver thrown to (the model does not see the coverage read, the route concept, the decoys or what the quarterback saw, so a gap marks a throw worth reviewing and is not evidence of a mistake):
```
                                              mean gap (yd)  median  share above 2 yd  share above 5 yd
expected yards, uncapped depth                        4.477   3.022             0.576             0.355
expected yards, depth credit capped at 20 yd          3.871   2.775             0.561             0.325
YAC opportunity only (p x E[YAC])                     2.180   1.501             0.427             0.137
```
The uncapped version is dominated by deep options: on 16.7% of throws its best option is more than 25 yd downfield, where the model has the least training data. Use the capped and YAC-only versions.
Does expected yards for the target predict what the throw gained? Correlation 0.481 (R2 0.232).
```
       n  predicted expected yards  actual mean yards
bin                                                  
0    681                      1.82               1.82
1    680                      3.62               3.97
2    680                      4.73               4.52
3    681                      5.77               5.54
4    680                      6.70               6.54
5    680                      7.71               7.41
6    681                      8.91               7.94
7    680                     10.41              10.05
8    680                     12.87              11.37
9    681                     19.88              19.34
```

## Limits
- Scope: the 6,805 thrown passes in plays.csv with a named target and no accepted penalty. plays.csv leaves out about 1,150 completions, mostly passes at or behind the line of scrimmage that gain more after the catch (mean 8.5 yd vs 4.4), and they have no tracking. Findings apply to passes thrown beyond the line.
- The catch is never tracked. The analysis model uses the first 0.5 s of ball flight and a constant-velocity projection to the estimated arrival; the decision model uses release only.
- Opportunity scores for receivers who were not thrown to extrapolate a model trained only on targeted receivers.
- Eight weeks of one season. Individual receivers and defenses do not have enough catches for stable ratings (see yac_probe.py and yac_defense_probe.py).
Run time 107s.