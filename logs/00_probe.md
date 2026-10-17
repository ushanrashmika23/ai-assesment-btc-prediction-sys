# Training smoke test (one epoch, nothing saved)

| | |
|---|---|
| command | `python src/train.py --arch gru --probe --epochs 1 --batch 1024` |
| exit code | `0` |
| started (UTC) | 2026-09-28 14:41:54 |
| finished (UTC) | 2026-09-28 14:43:25 |
| wall clock | 90.8s |
| python | 3.13.12 |
| platform | Windows-11-10.0.26200-SP0 |
| cwd | `E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction` |
| git | `not a git repository` |
| provenance | captured live by this invocation |

Confirms the trainer still runs against the rebuilt dataset and times one epoch before committing to the full run.

Output below is verbatim stdout+stderr from the command above, with only the
TensorFlow startup banner removed. Reproduce it by running the command shown.

## Output

```
python 3.13.12   tensorflow 2.21.0   keras 3.14.0
device: CPU only
threads: 10 intra-op, 5 inter-op

lookback 64   features 49   horizons ['15m', '30m', '45m', '60m', '75m', '90m']   HORIZON_MAX 6 candles
purge clearance: 1 candle(s)
using the full training split: 218,515 samples
val 46,824   test 46,686
regime head: ['15m', '30m', '60m', '90m'] (QUIET/ACTIVE) alongside 6 direction horizons
batches: 214 fit | 46 val | 46 test
direction class weights (strength 0.5), per horizon DOWN/NEUTRAL/UP:
   15m  1.047  0.907  1.047
   30m  1.047  0.907  1.047
   45m  1.047  0.907  1.047
   60m  1.047  0.907  1.047
   75m  1.047  0.907  1.047
   90m  1.047  0.907  1.047

PROBE - one epoch on the data above, nothing is saved.

214/214 - 78s - 365ms/step - direction_head_loss: 1.0982 - direction_head_sparse_categorical_accuracy: 0.4395 - loss: 3.0337 - regime_head_loss: 0.7098 - regime_head_sparse_categorical_accuracy: 0.5681 - return_head_loss: 1.2259 - return_head_mae: 0.7091 - val_direction_head_loss: 1.0301 - val_direction_head_sparse_categorical_accuracy: 0.4654 - val_loss: 2.0797 - val_regime_head_loss: 0.6166 - val_regime_head_sparse_categorical_accuracy: 0.6623 - val_return_head_loss: 0.4315 - val_return_head_mae: 0.4295

one epoch: 78.3s  (214 steps, 366 ms/step)
  + ~46 val batches of callback overhead per epoch
    5 epochs ->    6.5 min
   10 epochs ->   13.0 min
   25 epochs ->   32.6 min
   50 epochs ->   65.2 min

full split: 218,515 samples
```
