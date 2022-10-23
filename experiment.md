
# Experiment records  

- After 33th experiment, Transformer_E discard mean-pooling  
- After 33th experiment, RMSE calculation change to RMSE over all location and time step prediction

TODO: 45, 44 train with new config scheduler

|    | location  |     model      |  seq_base  |   input   | output | RMSE(TECU) | layer, hidd | norm, target |
|:--:|:---------:|:--------------:|:----------:|:---------:|:------:|:----------:|:-----------:|:------------:|
| 1  | (25, 120) |   LSTM (TEC)   |    time    |    0h     |  4hr   |   7.720    |             | min-max, yes |
| 2  | (25, 120) |   LSTM (TEC)   |    time    | -11hr~0hr |  4hr   |   4.435    |             | min-max, yes |
| 3  | (25, 120) |   LSTM (TEC)   |    time    | -23hr~0hr |  4hr   |   3.055    |             | min-max, yes |
| 4  | (25, 120) | LSTM (TEC+SW)  |    time    | -23hr~0hr |  4hr   |   3.048    |             | min-max, yes |
| 8  | (25, 120) |   LSTM (TEC)   |    time    | -23hr~0hr |  1hr   |   1.173    |             | min-max, yes |
| 9  | (25, 120) |   LSTM (TEC)   |    time    |    0h     |  4hr   |   7.753    |             | z-score, yes |
| 10 | (25, 120) |   LSTM (TEC)   |    time    | -11hr~0hr |  4hr   |   4.452    |             | z-score, yes |
| 11 | (25, 120) |   LSTM (TEC)   |    time    | -23hr~0hr |  4hr   |   3.220    |             | z-score, yes |
| 12 | (25, 120) |   LSTM (TEC)   |    time    |    0h     |  4hr   |   7.873    |             | z-score, no  |
| 13 | (25, 120) |   LSTM (TEC)   |    time    | -11hr~0hr |  4hr   |   8.808    |             | z-score, no  |
| 14 | (25, 120) |   LSTM (TEC)   |    time    | -23hr~0hr |  4hr   |   8.795    |             | z-score, no  |
| 24 |  global   | Transformer_E  |    time    | -23hr~0hr |  4hr   |   2.044    |   6, 512    | z-score, yes |
| 25 |  global   | Transformer_E  |    time    | -23hr~0hr |  4hr   |   2.095    |   6, 512    |  None, yes   |
| 26 |  global   | Transformer_E  |    time    | -23hr~0hr |  4hr   |   2.087    |      *      |   \*, yes    |
| 27 |  global   | Transformer_ED |    time    | -23hr~0hr |  4hr   |   4.090    |   1, 384    | z-score, yes |
| 28 |  global   | Transformer_ED |    time    | -23hr~0hr |  4hr   |   5.007    |    4, 64    | z-score, yes |
| 29 |  global   | Transformer_ED |    time    | -23hr~0hr |  4hr   |   5.178    |    6, 32    | z-score, yes |
| 30 |  global   | Transformer_ED |    time    | -23hr~0hr |  4hr   |   5.912    |   6, 1024   | z-score, yes |
| 31 |  global   | Transformer_ED |    time    | -23hr~0hr |  4hr   |   5.170    |    6, 16    | z-score, yes |
| 32 |  global   | Transformer_ED |    time    | -23hr~0hr |  4hr   |   4.746    |   2, 128    | z-score, yes |
| 33 |  global   | Transformer_ED |    time    | -23hr~0hr |  4hr   |   4.832    |   1, 128    | z-score, yes |
| 34 |  global   | Transformer_E  |    time    | -23hr~0hr |  4hr   |   1.907    |  12, 1024   | min-max, no  |
| 35 |  global   | Transformer_E  |    time    | -23hr~0hr |  4hr   |   1.479    |   12, 512   | min-max, no  |
| 36 |  global   | Transformer_E  |    time    | -23hr~0hr |  4hr   |   1.437    |   12, 256   | min-max, no  |
| 37 |  global   | Transformer_E  |    time    | -23hr~0hr |  4hr   |   1.432    |   12, 128   | min-max, no  |
| 38 |  global   | Transformer_E  |    time    | -23hr~0hr |  4hr   |   1.446    |   6, 128    | min-max, no  |
| 39 |  global   | Transformer_E  |    time    | -23hr~0hr |  4hr   |   1.475    |   12, 64    | min-max, no  |
| 40 |  global   | Transformer_E  |    time    | -23hr~0hr |  1hr   |   0.899    |   12, 128   | min-max, no  |
| 41 |  global   | Transformer_E  |  latitude  | -23hr~0hr |  1hr   |   0.718    |   12, 128   | min-max, no  |
| 42 |  global   | Transformer_E  |  latitude  | -23hr~0hr |  4hr   |   1.499    |   12, 128   | min-max, no  |
| 43 |  global   | Transformer_E  | longtitude | -23hr~0hr |  4hr   |   1.444    |   12, 128   | min-max, no  |
| 44 |  global   | Transformer_E  |  latitude  | -23hr~0hr |  1hr   |   0.551    |   12, 128   |     None     |
| 45 |  global   | Transformer_E  |    time    | -23hr~0hr |  1hr   |   0.704    |   12, 128   |     None     |
| 46 |  global   | Transformer_E  | longtitude | -23hr~0hr |  1hr   |   0.545    |   12, 128   |     None     |
| 47 |  global   | Transformer_E  | longtitude | -47hr~0hr |  1hr   |   68.675   |   12, 128   |     None     |
| 48 |  global   | Transformer_E  | longtitude | -47hr~0hr |  1hr   |   0.643    |   12, 128   | min-max, no  |
| 49 |  global   | Transformer_E  | longtitude | -11hr~0hr |  1hr   |   0.558    |   12, 128   |     None     |
| 50 |  global   | Transformer_E  | longtitude | -3hr~0hr  |  1hr   |   0.558    |   12, 128   |     None     |
| 51 |  global   | Transformer_E  | longtitude | -47hr~0hr |  1hr   |   0.662    |   6, 128    |     None     |
| 52 |  global   | Transformer_E  | longtitude | -23hr~0hr |  1hr   |   1.117    |   6, 128    |     None     |
| 53 |  global   | Transformer_E  | longtitude | -11hr~0hr |  1hr   |   0.944    |   6, 128    |     None     |
| 54 |  global   | Transformer_E  | longtitude | -3hr~0hr  |  1hr   |   0.549    |   6, 128    |     None     |
| 55 |  global   | Transformer_E  | longtitude | -47hr~0hr |  1hr   |   8.152    |   24, 128   |     None     |
| 56 |  global   | Transformer_E  | longtitude | -47hr~0hr |  1hr   |   0.682    |   24, 128   | min-max, no  |
| 57 |  global   | Transformer_E  | longtitude | -23hr~0hr |  1hr   |   0.553    |   24, 128   |     None     |
| 58 |  global   | Transformer_E  | longtitude | -11hr~0hr |  1hr   |   0.891    |   24, 128   |     None     |
| 59 |  global   | Transformer_E  | longtitude | -3hr~0hr  |  1hr   |   0.537    |   24, 128   |     None     |
| 60 |  global   | Transformer_E  |    time    | -23hr~0hr |  2hr   |   1.034    |   12, 128   |     None     |
| 61 |  global   | Transformer_E  |    time    | -23hr~0hr |  3hr   |   1.474    |   12, 128   |     None     |
| 62 |  global   | Transformer_E  |    time    | -23hr~0hr |  3hr   |   1.282    |   12, 128   | min-max, no  |
| 63 |  global   | Transformer_E  |    time    | -23hr~0hr |  6hr   |   3.336    |   12, 128   | min-max, no  |
| 64 |  global   | Transformer_E  |    time    | -23hr~0hr |  12hr  |   1.774    |   12, 128   | min-max, no  |
| 65 |  global   | Transformer_E  |    time    | -23hr~0hr |  24hr  |   1.769    |   12, 128   | min-max, no  |
| 66 |  global   | Transformer_E  |  latitude  | -23hr~0hr |  2hr   |   0.940    |   12, 128   |     None     |
| 67 |  global   | Transformer_E  |  latitude  | -23hr~0hr |  3hr   |   1.351    |   12, 128   | min-max, no  |
| 68 |  global   | Transformer_E  |  latitude  | -23hr~0hr |  6hr   |   1.694    |   12, 128   | min-max, no  |
| 69 |  global   | Transformer_E  |  latitude  | -23hr~0hr |  12hr  |   1.814    |   12, 128   | min-max, no  |
| 70 |  global   | Transformer_E  |  latitude  | -23hr~0hr |  24hr  |   1.888    |   12, 128   | min-max, no  |
| 71 |  global   | Transformer_E  | longtitude | -23hr~0hr |  2hr   |   0.967    |   12, 128   |     None     |
| 72 |  global   | Transformer_E  | longtitude | -23hr~0hr |  3hr   |   1.315    |   12, 128   | min-max, no  |
| 73 |  global   | Transformer_E  | longtitude | -23hr~0hr |  6hr   |   1.581    |   12, 128   | min-max, no  |
| 74 |  global   | Transformer_E  | longtitude | -23hr~0hr |  12hr  |   1.692    |   12, 128   | min-max, no  |
| 75 |  global   | Transformer_E  | longtitude | -23hr~0hr |  24hr  |   1.801    |   12, 128   | min-max, no  |

time (24, 71*72+2) (DOY, hour, SW)
lattitude (71, 24*72+3) (GTEC, DOY, hour, latitude)
longtitude (72, 24*71)

1. 輸入時間維度
2. 輸出時間維度
3. input feature (date, sin cos method)