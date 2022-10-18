
# Experiment records  

- After 33th experiment, Transformer_E discard mean-pooling  

<details>
<summary> details </summary>


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
| 34 |  global   | Transformer_E  |    time    | -23hr~0hr |  4hr   |            |  12, 1024   | z-score, no  |
| 35 |  global   | Transformer_E  |    time    | -23hr~0hr |  4hr   |            |   12, 512   | z-score, no  |
| 36 |  global   | Transformer_E  |    time    | -23hr~0hr |  4hr   |            |   12, 256   | z-score, no  |
| 37 |  global   | Transformer_E  |    time    | -23hr~0hr |  4hr   |   1.432    |   12, 128   | z-score, no  |
| 38 |  global   | Transformer_E  |    time    | -23hr~0hr |  4hr   |            |   6, 128    | z-score, no  |
| 39 |  global   | Transformer_E  |    time    | -23hr~0hr |  4hr   |            |   12, 64    | z-score, no  |
| 40 |  global   | Transformer_E  |    time    | -23hr~0hr |  1hr   |   0.899    |   12, 128   | z-score, no  |
| 41 |  global   | Transformer_E  |  latitude  | -23hr~0hr |  1hr   |   0.718    |   12, 128   | z-score, no  |
| 42 |  global   | Transformer_E  |  latitude  | -23hr~0hr |  4hr   |   1.499    |   12, 128   | z-score, no  |
| 43 |  global   | Transformer_E  | longtitude | -23hr~0hr |  4hr   |   1.444    |   12, 128   | z-score, no  |
</details>