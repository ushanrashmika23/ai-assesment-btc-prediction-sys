# Step 3 - Evaluation (validation + test)

| | |
|---|---|
| command | `python src/evaluate.py --split both` |
| exit code | `0` |
| started (UTC) | 2026-09-28 15:42:06 |
| finished (UTC) | 2026-09-28 15:43:04 |
| wall clock | 58.0s |
| python | 3.13.12 |
| platform | Windows-11-10.0.26200-SP0 |
| cwd | `E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction` |
| git | `e7cf297` |
| provenance | captured live by this invocation |

The trained 3-seed GRU ensemble, evaluated on the held-out validation and test splits. Read-only: this script selects nothing.

Output below is verbatim stdout+stderr from the command above, with only the
TensorFlow startup banner removed. Reproduce it by running the command shown.

## Output

```
inference_bundle.json declares an ensemble of 3 members - evaluating it as ONE model.
use --no-ensemble to evaluate a single member instead.
tensorflow 2.21.0  |  keras 3.14.0  |  device: CPU
artefacts: 218,515 train / 46,824 val / 46,686 test   |   purge clearance 1 candle(s)
scaler and thresholds are LOADED from data/final/, never refitted.

 NOTE ON THE TEST SPLIT
   This report is read-only; it selects nothing. Model choice
   belongs to validation macro-F1 (src/train.py). If you change
   the model because of a number below and re-run, the test
   split has become a second validation split.
================================================================================
 MODEL EVALUATION
================================================================================
 model            ENSEMBLE of 3 members
                    - gru_s42_best.keras  (641 KB)
                    - gru_s43_best.keras  (641 KB)
                    - gru_s44_best.keras  (641 KB)
 combine          mean of per-member softmax probabilities
 parameters       147,744 (per member)
 architecture     GRU
 trained (UTC)    2026-09-28T15:35:45.040314+00:00
 trained by       src/train.py
 seed             42
 lookback         64
 train samples    218515
 subset used      False
 chosen on        val_selection_score  {'gru': 0.5504903887395078}

================================================================================
 VALIDATION  -  46,824 samples  -  2024-01-15 -> 2025-05-17
================================================================================

 REGIME  (QUIET vs ACTIVE - THE ACCURACY HEADLINE)
   'will the mean |15m return| over the next h candles exceed its
    own trailing 24h average?'  Both classes are ~50% by
    construction, so 0.50 IS the baseline and the lift is real.
     majority_acc  model_acc   lift  model_macroF1  base_rate
15m        0.4936     0.6186 0.1249         0.6181     0.5064
30m        0.4931     0.6606 0.1675         0.6602     0.5069
60m        0.4996     0.6963 0.1967         0.6962     0.5004
90m        0.4976     0.7091 0.2115         0.7087     0.5024
    15m  accuracy 0.6186  ##############........  +0.1249 vs baseline
    30m  accuracy 0.6606  ###############.......  +0.1675 vs baseline
    60m  accuracy 0.6963  ###############.......  +0.1967 vs baseline
    90m  accuracy 0.7091  ################......  +0.2115 vs baseline

 REGIME PER-CLASS DETAIL
     QUIET_prec  QUIET_rec  QUIET_f1  ACTIVE_prec  ACTIVE_rec  ACTIVE_f1
15m      0.6188     0.5920    0.6051       0.6184      0.6444     0.6311
30m      0.6614     0.6384    0.6497       0.6598      0.6821     0.6708
60m      0.7017     0.6821    0.6918       0.6912      0.7105     0.7007
90m      0.7227     0.6741    0.6976       0.6973      0.7437     0.7198

 REGIME CONFUSION  (rows = actual, cols = predicted)

  15m horizon
    actual \ pred        QUIET      ACTIVE    recall
    QUIET          13,684 ( 59%)  9,430 ( 41%)     59.2%
    ACTIVE          8,431 ( 36%) 15,279 ( 64%)     64.4%
    precision: QUIET 61.9%   ACTIVE 61.8%

  30m horizon
    actual \ pred        QUIET      ACTIVE    recall
    QUIET          14,739 ( 64%)  8,349 ( 36%)     63.8%
    ACTIVE          7,545 ( 32%) 16,191 ( 68%)     68.2%
    precision: QUIET 66.1%   ACTIVE 66.0%

  60m horizon
    actual \ pred        QUIET      ACTIVE    recall
    QUIET          15,957 ( 68%)  7,438 ( 32%)     68.2%
    ACTIVE          6,783 ( 29%) 16,646 ( 71%)     71.0%
    precision: QUIET 70.2%   ACTIVE 69.1%

  90m horizon
    actual \ pred        QUIET      ACTIVE    recall
    QUIET          15,708 ( 67%)  7,593 ( 33%)     67.4%
    ACTIVE          6,028 ( 26%) 17,495 ( 74%)     74.4%
    precision: QUIET 72.3%   ACTIVE 69.7%

 REGIME SELECTIVE ACCURACY  (answer only above a confidence floor)
      accuracy  coverage  n_answered
0.50    0.6711    1.0000 46,824.0000
0.55    0.7040    0.8034 37,618.0000
0.60    0.7367    0.6236 29,200.0000
0.65    0.7716    0.4624 21,650.0000
0.70    0.8080    0.3213 15,046.0000
0.80    0.8688    0.1144  5,356.0000
0.90    0.9366    0.0140    655.0000
0.95    1.0000    0.0005     22.0000
   -> choose a floor from THIS column (it is a tunable).
   -> coverage is the share of calls you would still make; the trade is
      explicit: higher floor, higher accuracy, fewer answers.

 DIRECTION  (the harder question - reported on macro-F1, not accuracy)

 REGRESSION  (percent return; lower MAE is better)
     naive_mean_MAE  model_MAE     delta  model_RMSE      bias     corr
15m        0.184367   0.184361 -0.000007    0.279608 -0.003448 0.030173
30m        0.256909   0.257824  0.000915    0.396788 -0.010568 0.034132
45m        0.312483   0.314408  0.001925    0.485091 -0.013839 0.039597
60m        0.359013   0.362207  0.003194    0.558707 -0.018835 0.041319
75m        0.400595   0.404483  0.003888    0.622600 -0.013202 0.042084
90m        0.438755   0.443907  0.005152    0.681032 -0.018901 0.044416
   -> worse than the mean at 5/6 horizons.
     majority_acc  model_acc  majority_macroF1  model_macroF1   lift      n
15m        0.4353     0.4833            0.2022         0.4351 2.1523  46824
30m        0.4375     0.4757            0.2029         0.4312 2.1251  46824
45m        0.4362     0.4734            0.2025         0.4308 2.1277  46824
60m        0.4314     0.4673            0.2009         0.4282 2.1311  46824
75m        0.4323     0.4655            0.2012         0.4275 2.1244  46824
90m        0.4300     0.4618            0.2005         0.4264 2.1273  46824
    15m  macro-F1 0.4351  ##########............  beats baseline 0.2022
    30m  macro-F1 0.4312  #########.............  beats baseline 0.2029
    45m  macro-F1 0.4308  #########.............  beats baseline 0.2025
    60m  macro-F1 0.4282  #########.............  beats baseline 0.2009
    75m  macro-F1 0.4275  #########.............  beats baseline 0.2012
    90m  macro-F1 0.4264  #########.............  beats baseline 0.2005

 PER-CLASS DETAIL  (precision / recall / F1)
     DOWN_prec  DOWN_rec  DOWN_f1  NEUTRAL_prec  NEUTRAL_rec  NEUTRAL_f1  UP_prec  UP_rec  UP_f1
15m     0.3747    0.3218   0.3462        0.5587       0.7203      0.6293   0.4021  0.2797 0.3299
30m     0.3673    0.3401   0.3532        0.5550       0.6930      0.6164   0.3977  0.2732 0.3239
45m     0.3648    0.3487   0.3566        0.5521       0.6842      0.6111   0.4034  0.2717 0.3247
60m     0.3577    0.3448   0.3511        0.5449       0.6724      0.6020   0.4088  0.2787 0.3314
75m     0.3533    0.3422   0.3477        0.5461       0.6654      0.5999   0.4066  0.2845 0.3348
90m     0.3564    0.3675   0.3619        0.5470       0.6497      0.5939   0.3976  0.2726 0.3235
   -> every class is predicted at least sometimes (recall > 5% somewhere).
   -> low DOWN/UP recall with high NEUTRAL recall means the model mostly says NEUTRAL.

 CONFUSION  (rows = actual, cols = predicted; counts and row %)

  15m horizon
    actual \ pred         DOWN     NEUTRAL          UP    recall
    DOWN            4,242 ( 32%)  5,837 ( 44%)  3,102 ( 24%)     32.2%
    NEUTRAL         3,288 ( 16%) 14,681 ( 72%)  2,412 ( 12%)     72.0%
    UP              3,792 ( 29%)  5,761 ( 43%)  3,709 ( 28%)     28.0%
    precision: DOWN 37.5%   NEUTRAL 55.9%   UP 40.2%

  30m horizon
    actual \ pred         DOWN     NEUTRAL          UP    recall
    DOWN            4,480 ( 34%)  5,774 ( 44%)  2,917 ( 22%)     34.0%
    NEUTRAL         3,755 ( 18%) 14,196 ( 69%)  2,533 ( 12%)     69.3%
    UP              3,962 ( 30%)  5,609 ( 43%)  3,598 ( 27%)     27.3%
    precision: DOWN 36.7%   NEUTRAL 55.5%   UP 39.8%

  45m horizon
    actual \ pred         DOWN     NEUTRAL          UP    recall
    DOWN            4,604 ( 35%)  5,809 ( 44%)  2,791 ( 21%)     34.9%
    NEUTRAL         3,936 ( 19%) 13,975 ( 68%)  2,513 ( 12%)     68.4%
    UP              4,080 ( 31%)  5,530 ( 42%)  3,586 ( 27%)     27.2%
    precision: DOWN 36.5%   NEUTRAL 55.2%   UP 40.3%

  60m horizon
    actual \ pred         DOWN     NEUTRAL          UP    recall
    DOWN            4,580 ( 34%)  5,820 ( 44%)  2,884 ( 22%)     34.5%
    NEUTRAL         4,123 ( 20%) 13,582 ( 67%)  2,494 ( 12%)     67.2%
    UP              4,101 ( 31%)  5,522 ( 41%)  3,718 ( 28%)     27.9%
    precision: DOWN 35.8%   NEUTRAL 54.5%   UP 40.9%

  75m horizon
    actual \ pred         DOWN     NEUTRAL          UP    recall
    DOWN            4,537 ( 34%)  5,747 ( 43%)  2,976 ( 22%)     34.2%
    NEUTRAL         4,216 ( 21%) 13,468 ( 67%)  2,557 ( 13%)     66.5%
    UP              4,087 ( 31%)  5,445 ( 41%)  3,791 ( 28%)     28.5%
    precision: DOWN 35.3%   NEUTRAL 54.6%   UP 40.7%

  90m horizon
    actual \ pred         DOWN     NEUTRAL          UP    recall
    DOWN            4,900 ( 37%)  5,571 ( 42%)  2,862 ( 21%)     36.8%
    NEUTRAL         4,396 ( 22%) 13,080 ( 65%)  2,656 ( 13%)     65.0%
    UP              4,454 ( 33%)  5,263 ( 39%)  3,642 ( 27%)     27.3%
    precision: DOWN 35.6%   NEUTRAL 54.7%   UP 39.8%

 CALIBRATION  (is the confidence meaningful?)
                  n  share  mean_conf  accuracy     gap
bucket                                                 
[0.33, 0.40)  78310 0.2787     0.3752    0.3682 -0.0070
[0.40, 0.45)  76627 0.2727     0.4235    0.4118 -0.0117
[0.45, 0.50)  45102 0.1605     0.4722    0.4549 -0.0172
[0.50, 0.55)  24525 0.0873     0.5228    0.5243  0.0015
[0.55, 0.60)  16158 0.0575     0.5738    0.5852  0.0115
[0.60, 0.70)  21398 0.0762     0.6457    0.6594  0.0137
[0.70, 1.01)  18824 0.0670     0.7805    0.7987  0.0182
   -> accuracy rises with confidence: the probability output carries information.
   -> 'gap' = accuracy - mean confidence. Negative means overconfident.

 CONFIDENCE FLOOR SWEEP  (what the dashboard's 'c' key buys you)
      macro_F1  accuracy  directional_share
0.00    0.4299    0.4712             0.4637
0.34    0.4294    0.4711             0.4611
0.38    0.4084    0.4728             0.3608
0.42    0.3536    0.4674             0.2039
0.46    0.2849    0.4536             0.0878
0.50    0.2364    0.4429             0.0303
0.55    0.2108    0.4364             0.0070
0.60    0.2044    0.4347             0.0018
0.70    0.2021    0.4339             0.0002
   always-NEUTRAL accuracy on this split: 0.4338
   -> THIS is the column to choose a floor from. It is a tunable.
   -> floors that beat always-NEUTRAL on accuracy: 0.00, 0.34, 0.38, 0.42, 0.46, 0.50, 0.55, 0.60, 0.70
   -> commitment falls from 46.4% of calls to 0.0% across the sweep.

================================================================================
 TEST  -  46,686 samples  -  2025-05-18 -> 2026-09-16
================================================================================

 REGIME  (QUIET vs ACTIVE - THE ACCURACY HEADLINE)
   'will the mean |15m return| over the next h candles exceed its
    own trailing 24h average?'  Both classes are ~50% by
    construction, so 0.50 IS the baseline and the lift is real.
     majority_acc  model_acc   lift  model_macroF1  base_rate
15m        0.5039     0.6004 0.0964         0.5985     0.5039
30m        0.4975     0.6400 0.1425         0.6381     0.5025
60m        0.4951     0.6708 0.1757         0.6692     0.5049
90m        0.4942     0.6799 0.1856         0.6773     0.5058
    15m  accuracy 0.6004  #############.........  +0.0964 vs baseline
    30m  accuracy 0.6400  ##############........  +0.1425 vs baseline
    60m  accuracy 0.6708  ###############.......  +0.1757 vs baseline
    90m  accuracy 0.6799  ###############.......  +0.1856 vs baseline

 REGIME PER-CLASS DETAIL
     QUIET_prec  QUIET_rec  QUIET_f1  ACTIVE_prec  ACTIVE_rec  ACTIVE_f1
15m      0.6216     0.5288    0.5715       0.5844      0.6730     0.6256
30m      0.6598     0.5705    0.6119       0.6250      0.7088     0.6643
60m      0.6899     0.6087    0.6467       0.6560      0.7317     0.6918
90m      0.7087     0.5980    0.6487       0.6592      0.7599     0.7060

 REGIME CONFUSION  (rows = actual, cols = predicted)

  15m horizon
    actual \ pred        QUIET      ACTIVE    recall
    QUIET          12,441 ( 53%) 11,086 ( 47%)     52.9%
    ACTIVE          7,572 ( 33%) 15,587 ( 67%)     67.3%
    precision: QUIET 62.2%   ACTIVE 58.4%

  30m horizon
    actual \ pred        QUIET      ACTIVE    recall
    QUIET          13,251 ( 57%)  9,976 ( 43%)     57.0%
    ACTIVE          6,832 ( 29%) 16,627 ( 71%)     70.9%
    precision: QUIET 66.0%   ACTIVE 62.5%

  60m horizon
    actual \ pred        QUIET      ACTIVE    recall
    QUIET          14,069 ( 61%)  9,045 ( 39%)     60.9%
    ACTIVE          6,325 ( 27%) 17,247 ( 73%)     73.2%
    precision: QUIET 69.0%   ACTIVE 65.6%

  90m horizon
    actual \ pred        QUIET      ACTIVE    recall
    QUIET          13,797 ( 60%)  9,276 ( 40%)     59.8%
    ACTIVE          5,670 ( 24%) 17,943 ( 76%)     76.0%
    precision: QUIET 70.9%   ACTIVE 65.9%

 REGIME SELECTIVE ACCURACY  (answer only above a confidence floor)
      accuracy  coverage  n_answered
0.50    0.6477    1.0000 46,686.0000
0.55    0.6751    0.8082 37,734.0000
0.60    0.7053    0.6240 29,132.0000
0.65    0.7382    0.4587 21,416.0000
0.70    0.7725    0.3174 14,819.0000
0.80    0.8493    0.1148  5,358.0000
0.90    0.9412    0.0148    691.0000
0.95    0.9938    0.0005     24.0000
   -> NOT a column to choose a floor from - that would be tuning on test.
   -> coverage is the share of calls you would still make; the trade is
      explicit: higher floor, higher accuracy, fewer answers.

 DIRECTION  (the harder question - reported on macro-F1, not accuracy)

 REGRESSION  (percent return; lower MAE is better)
     naive_mean_MAE  model_MAE    delta  model_RMSE      bias      corr
15m        0.145945   0.146489 0.000544    0.226287 -0.001392  0.012205
30m        0.205102   0.207241 0.002139    0.318735 -0.010209  0.011904
45m        0.249220   0.252593 0.003373    0.389465 -0.010538  0.007239
60m        0.287304   0.292507 0.005203    0.448928 -0.014533  0.003779
75m        0.321510   0.327982 0.006472    0.501135 -0.009170  0.001268
90m        0.352292   0.361520 0.009228    0.549594 -0.015869 -0.004587
   -> WORSE than predicting the mean at every horizon.
      The regression head has no usable skill. Do not read the
      'expected' number in the dashboard as a forecast.
     majority_acc  model_acc  majority_macroF1  model_macroF1   lift      n
15m        0.5383     0.5450            0.2333         0.4279 1.8342  46686
30m        0.5221     0.5356            0.2287         0.4276 1.8699  46686
45m        0.5190     0.5299            0.2278         0.4250 1.8660  46686
60m        0.5125     0.5217            0.2259         0.4194 1.8567  46686
75m        0.5148     0.5204            0.2266         0.4211 1.8584  46686
90m        0.5144     0.5163            0.2264         0.4212 1.8600  46686
    15m  macro-F1 0.4279  #########.............  beats baseline 0.2333
    30m  macro-F1 0.4276  #########.............  beats baseline 0.2287
    45m  macro-F1 0.4250  #########.............  beats baseline 0.2278
    60m  macro-F1 0.4194  #########.............  beats baseline 0.2259
    75m  macro-F1 0.4211  #########.............  beats baseline 0.2266
    90m  macro-F1 0.4212  #########.............  beats baseline 0.2264

 PER-CLASS DETAIL  (precision / recall / F1)
     DOWN_prec  DOWN_rec  DOWN_f1  NEUTRAL_prec  NEUTRAL_rec  NEUTRAL_f1  UP_prec  UP_rec  UP_f1
15m     0.3586    0.2422   0.2892        0.6322       0.8052      0.7083   0.3513  0.2414 0.2861
30m     0.3591    0.2554   0.2985        0.6174       0.8014      0.6975   0.3693  0.2345 0.2868
45m     0.3589    0.2678   0.3067        0.6121       0.7922      0.6906   0.3651  0.2242 0.2778
60m     0.3555    0.2676   0.3054        0.6041       0.7870      0.6835   0.3573  0.2162 0.2694
75m     0.3530    0.2727   0.3077        0.6058       0.7760      0.6804   0.3578  0.2235 0.2751
90m     0.3464    0.2934   0.3177        0.6089       0.7613      0.6766   0.3542  0.2171 0.2692
   -> every class is predicted at least sometimes (recall > 5% somewhere).
   -> low DOWN/UP recall with high NEUTRAL recall means the model mostly says NEUTRAL.

 CONFUSION  (rows = actual, cols = predicted; counts and row %)

  15m horizon
    actual \ pred         DOWN     NEUTRAL          UP    recall
    DOWN            2,679 ( 24%)  6,146 ( 56%)  2,234 ( 20%)     24.2%
    NEUTRAL         2,451 ( 10%) 20,233 ( 81%)  2,445 ( 10%)     80.5%
    UP              2,340 ( 22%)  5,624 ( 54%)  2,534 ( 24%)     24.1%
    precision: DOWN 35.9%   NEUTRAL 63.2%   UP 35.1%

  30m horizon
    actual \ pred         DOWN     NEUTRAL          UP    recall
    DOWN            2,944 ( 26%)  6,410 ( 56%)  2,171 ( 19%)     25.5%
    NEUTRAL         2,694 ( 11%) 19,534 ( 80%)  2,148 (  9%)     80.1%
    UP              2,561 ( 24%)  5,695 ( 53%)  2,529 ( 23%)     23.4%
    precision: DOWN 35.9%   NEUTRAL 61.7%   UP 36.9%

  45m horizon
    actual \ pred         DOWN     NEUTRAL          UP    recall
    DOWN            3,113 ( 27%)  6,393 ( 55%)  2,119 ( 18%)     26.8%
    NEUTRAL         2,931 ( 12%) 19,197 ( 79%)  2,103 (  9%)     79.2%
    UP              2,629 ( 24%)  5,773 ( 53%)  2,428 ( 22%)     22.4%
    precision: DOWN 35.9%   NEUTRAL 61.2%   UP 36.5%

  60m horizon
    actual \ pred         DOWN     NEUTRAL          UP    recall
    DOWN            3,160 ( 27%)  6,520 ( 55%)  2,127 ( 18%)     26.8%
    NEUTRAL         2,965 ( 12%) 18,829 ( 79%)  2,132 (  9%)     78.7%
    UP              2,765 ( 25%)  5,820 ( 53%)  2,368 ( 22%)     21.6%
    precision: DOWN 35.5%   NEUTRAL 60.4%   UP 35.7%

  75m horizon
    actual \ pred         DOWN     NEUTRAL          UP    recall
    DOWN            3,225 ( 27%)  6,464 ( 55%)  2,136 ( 18%)     27.3%
    NEUTRAL         3,179 ( 13%) 18,651 ( 78%)  2,206 (  9%)     77.6%
    UP              2,733 ( 25%)  5,673 ( 52%)  2,419 ( 22%)     22.3%
    precision: DOWN 35.3%   NEUTRAL 60.6%   UP 35.8%

  90m horizon
    actual \ pred         DOWN     NEUTRAL          UP    recall
    DOWN            3,466 ( 29%)  6,259 ( 53%)  2,089 ( 18%)     29.3%
    NEUTRAL         3,523 ( 15%) 18,282 ( 76%)  2,208 (  9%)     76.1%
    UP              3,016 ( 28%)  5,486 ( 51%)  2,357 ( 22%)     21.7%
    precision: DOWN 34.6%   NEUTRAL 60.9%   UP 35.4%

 CALIBRATION  (is the confidence meaningful?)
                  n  share  mean_conf  accuracy     gap
bucket                                                 
[0.33, 0.40)  68454 0.2444     0.3746    0.3667 -0.0079
[0.40, 0.45)  59524 0.2125     0.4233    0.4290  0.0057
[0.45, 0.50)  37260 0.1330     0.4731    0.4916  0.0185
[0.50, 0.55)  26415 0.0943     0.5238    0.5487  0.0250
[0.55, 0.60)  20439 0.0730     0.5740    0.6087  0.0347
[0.60, 0.70)  29678 0.1059     0.6476    0.6844  0.0368
[0.70, 1.01)  38346 0.1369     0.7924    0.8279  0.0355
   -> accuracy rises with confidence: the probability output carries information.
   -> 'gap' = accuracy - mean confidence. Negative means overconfident.

 CONFIDENCE FLOOR SWEEP  (what the dashboard's 'c' key buys you)
      macro_F1  accuracy  directional_share
0.00    0.4237    0.5282             0.3325
0.34    0.4232    0.5284             0.3304
0.38    0.3960    0.5337             0.2402
0.42    0.3378    0.5347             0.1200
0.46    0.2793    0.5281             0.0472
0.50    0.2469    0.5234             0.0158
0.55    0.2329    0.5214             0.0033
0.60    0.2293    0.5205             0.0009
0.70    0.2283    0.5202             0.0002
   always-NEUTRAL accuracy on this split: 0.5202
   -> Do NOT choose a floor from this. Choosing on test would be tuning on test.
   -> floors that beat always-NEUTRAL on accuracy: 0.00, 0.34, 0.38, 0.42, 0.46, 0.50, 0.55, 0.60, 0.70
   -> commitment falls from 33.2% of calls to 0.0% across the sweep.

================================================================================
 VERDICT
================================================================================
 ensemble_x3 [VALIDATION]
   regime head    : beats the majority-class accuracy at 4/4 horizons, mean lift +0.1751 - THE HEADLINE
   direction head : beats the majority-class macro-F1 at 6/6 horizons
   regression head: worse than the mean at 5/6 horizons
 ensemble_x3 [TEST]
   regime head    : beats the majority-class accuracy at 4/4 horizons, mean lift +0.1501 - THE HEADLINE
   direction head : beats the majority-class macro-F1 at 6/6 horizons
   regression head: WORSE than predicting the mean at 6/6 horizons - no usable skill

 Regime accuracy is not profitability either. 'This window will be more volatile
 than the last 24 hours' is not a trade: there is no position, no backtest, no
 fees and no slippage here, and the test period is one regime. Nothing above is
 a promise about the future.
 wrote E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction\models\eval_summary.json
```
