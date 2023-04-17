
# Experiment records  

- After 33th experiment, Transformer_E discard mean-pooling  
- After 33th experiment, RMSE calculation change to RMSE over all location and time step prediction

|       | location  |     model      | features | seq_base  |   input   |  output  | RMSE(set1U) | layer, hidd | norm, target |
| :---: | :-------: | :------------: | :------: | :-------: | :-------: | :------: | :---------: | :---------: | :----------: |
|   1   | (25, 120) |      LSTM      |   set1   |   time    |    0h     |   4hr    |    7.720    |             | min-max, yes |
|   2   | (25, 120) |      LSTM      |   set1   |   time    | -11hr~0hr |   4hr    |    4.435    |             | min-max, yes |
|   3   | (25, 120) |      LSTM      |   set1   |   time    | -23hr~0hr |   4hr    |    3.055    |             | min-max, yes |
|   4   | (25, 120) |      LSTM      |   set2   |   time    | -23hr~0hr |   4hr    |    3.048    |             | min-max, yes |
|   8   | (25, 120) |      LSTM      |   set1   |   time    | -23hr~0hr |   1hr    |    1.173    |             | min-max, yes |
|   9   | (25, 120) |      LSTM      |   set1   |   time    |    0h     |   4hr    |    7.753    |             | z-score, yes |
|  10   | (25, 120) |      LSTM      |   set1   |   time    | -11hr~0hr |   4hr    |    4.452    |             | z-score, yes |
|  11   | (25, 120) |      LSTM      |   set1   |   time    | -23hr~0hr |   4hr    |    3.220    |             | z-score, yes |
|  12   | (25, 120) |      LSTM      |   set1   |   time    |    0h     |   4hr    |    7.873    |             | z-score, no  |
|  13   | (25, 120) |      LSTM      |   set1   |   time    | -11hr~0hr |   4hr    |    8.808    |             | z-score, no  |
|  14   | (25, 120) |      LSTM      |   set1   |   time    | -23hr~0hr |   4hr    |    8.795    |             | z-score, no  |
|  24   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   4hr    |    2.044    |   6, 512    | z-score, yes |
|  25   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   4hr    |    2.095    |   6, 512    |  None, yes   |
|  26   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   4hr    |    2.087    |      *      |   \*, yes    |
|  27   |  global   | Transformer_ED |   set1   |   time    | -23hr~0hr |   4hr    |    4.090    |   1, 384    | z-score, yes |
|  28   |  global   | Transformer_ED |   set1   |   time    | -23hr~0hr |   4hr    |    5.007    |    4, 64    | z-score, yes |
|  29   |  global   | Transformer_ED |   set1   |   time    | -23hr~0hr |   4hr    |    5.178    |    6, 32    | z-score, yes |
|  30   |  global   | Transformer_ED |   set1   |   time    | -23hr~0hr |   4hr    |    5.912    |   6, 1024   | z-score, yes |
|  31   |  global   | Transformer_ED |   set1   |   time    | -23hr~0hr |   4hr    |    5.170    |    6, 16    | z-score, yes |
|  32   |  global   | Transformer_ED |   set1   |   time    | -23hr~0hr |   4hr    |    4.746    |   2, 128    | z-score, yes |
|  33   |  global   | Transformer_ED |   set1   |   time    | -23hr~0hr |   4hr    |    4.832    |   1, 128    | z-score, yes |
|  34   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   4hr    |    1.907    |  12, 1024   | min-max, no  |
|  35   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   4hr    |    1.479    |   12, 512   | min-max, no  |
|  36   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   4hr    |    1.437    |   12, 256   | min-max, no  |
|  37   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   4hr    |    1.432    |   12, 128   | min-max, no  |
|  38   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   4hr    |    1.446    |   6, 128    | min-max, no  |
|  39   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   4hr    |    1.475    |   12, 64    | min-max, no  |
|  40   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   1hr    |    0.893    |   12, 128   | min-max, no  |
|  41   |  global   | Transformer_E  |   set1   | latitude  | -23hr~0hr |   1hr    |    0.619    |   12, 128   | min-max, no  |
|  42   |  global   | Transformer_E  |   set1   | latitude  | -23hr~0hr |   4hr    |    1.499    |   12, 128   | min-max, no  |
|  43   |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   4hr    |    1.444    |   12, 128   | min-max, no  |
|  44   |  global   | Transformer_E  |   set1   | latitude  | -23hr~0hr |   1hr    |    0.551    |   12, 128   |     None     |
|  45   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   1hr    |    0.704    |   12, 128   |     None     |
|  46   |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    0.545    |   12, 128   |     None     |
|  47   |  global   | Transformer_E  |   set1   | longitude | -47hr~0hr |   1hr    |   68.675    |   12, 128   |     None     |
|  48   |  global   | Transformer_E  |   set1   | longitude | -47hr~0hr |   1hr    |    0.643    |   12, 128   | min-max, no  |
|  49   |  global   | Transformer_E  |   set1   | longitude | -11hr~0hr |   1hr    |    0.558    |   12, 128   | min-max, no  |
|  50   |  global   | Transformer_E  |   set1   | longitude | -3hr~0hr  |   1hr    |    0.558    |   12, 128   | min-max, no  |
|  51   |  global   | Transformer_E  |   set1   | longitude | -47hr~0hr |   1hr    |    0.662    |   6, 128    |     None     |
|  52   |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    0.550    |   6, 128    | min-max, no  |
|  53   |  global   | Transformer_E  |   set1   | longitude | -11hr~0hr |   1hr    |    0.591    |   6, 128    | min-max, no  |
|  54   |  global   | Transformer_E  |   set1   | longitude | -3hr~0hr  |   1hr    |    0.549    |   6, 128    |     None     |
|  55   |  global   | Transformer_E  |   set1   | longitude | -47hr~0hr |   1hr    |    8.152    |   24, 128   |     None     |
|  56   |  global   | Transformer_E  |   set1   | longitude | -47hr~0hr |   1hr    |    0.682    |   24, 128   | min-max, no  |
|  57   |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    0.553    |   24, 128   |     None     |
|  58   |  global   | Transformer_E  |   set1   | longitude | -11hr~0hr |   1hr    |    0.913    |   24, 128   | min-max, no  |
|  59   |  global   | Transformer_E  |   set1   | longitude | -3hr~0hr  |   1hr    |    0.537    |   24, 128   |     None     |
|  60   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   2hr    |    1.034    |   12, 128   |     None     |
|  61   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   3hr    |    1.270    |   12, 128   | min-max, no  |
|  62   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   3hr    |    1.282    |   12, 128   | min-max, no  |
|  63   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   6hr    |    1.616    |   12, 128   | min-max, no  |
|  64   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   12hr   |    1.774    |   12, 128   | min-max, no  |
|  65   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   24hr   |    1.769    |   12, 128   | min-max, no  |
|  66   |  global   | Transformer_E  |   set1   | latitude  | -23hr~0hr |   2hr    |    0.940    |   12, 128   |     None     |
|  67   |  global   | Transformer_E  |   set1   | latitude  | -23hr~0hr |   3hr    |    1.351    |   12, 128   | min-max, no  |
|  68   |  global   | Transformer_E  |   set1   | latitude  | -23hr~0hr |   6hr    |    1.694    |   12, 128   | min-max, no  |
|  69   |  global   | Transformer_E  |   set1   | latitude  | -23hr~0hr |   12hr   |    1.814    |   12, 128   | min-max, no  |
|  70   |  global   | Transformer_E  |   set1   | latitude  | -23hr~0hr |   24hr   |    1.888    |   12, 128   | min-max, no  |
|  71   |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   2hr    |    0.967    |   12, 128   |     None     |
|  72   |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   3hr    |    1.315    |   12, 128   | min-max, no  |
|  73   |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   6hr    |    1.581    |   12, 128   | min-max, no  |
|  74   |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   12hr   |    1.692    |   12, 128   | min-max, no  |
|  75   |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   24hr   |    1.801    |   12, 128   | min-max, no  |
|  76   |  global   | Transformer_E  |   set1   | longitude | -47hr~0hr |   4hr    |    1.767    |   12, 128   | min-max, no  |
|  77   |  global   | Transformer_E  |   set1   | longitude | -11hr~0hr |   4hr    |    2.250    |   12, 128   | min-max, no  |
|  78   |  global   | Transformer_E  |   set1   | longitude | -3hr~0hr  |   4hr    |    2.307    |   12, 128   | min-max, no  |
|  79   |  global   | Transformer_E  |   set1   | latitude  | -3hr~0hr  |   4hr    |    2.775    |   12, 128   | min-max, no  |
|  80   |  global   | Transformer_E  |   set1   | longitude | -47hr~0hr |   4hr    |    1.644    |   12, 256   | min-max, no  |
|  81   |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   4hr    |    1.448    |   12, 256   | min-max, no  |
|  82   |  global   | Transformer_E  |   set1   | longitude | -11hr~0hr |   4hr    |    2.029    |   12, 256   | min-max, no  |
|  83   |  global   | Transformer_E  |   set1   | longitude | -3hr~0hr  |   4hr    |    2.096    |   12, 256   | min-max, no  |
|  84   |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   48hr   |    1.907    |   12, 128   | min-max, no  |
|  85   |  global   | Transformer_E  |   set1   | latitude  | -23hr~0hr |   48hr   |    2.075    |   12, 128   | min-max, no  |
|  86   |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   48hr   |    1.972    |   12, 128   | min-max, no  |
|  87   |  global   | Transformer_ED |   set1   |   time    | -23hr~0hr | 1hr~24hr |             |   12, 256   | min-max, no  |
|  88   |  global   | Transformer_ED |   set1   |   time    | -23hr~0hr | 1hr~4hr  |    2.746    |   6, 128    | min-max, no  |
|  89   |  global   | Transformer_E  |   set3   | longitude | -23hr~0hr |   1hr    |    0.584    |   12, 128   | min-max, no  |
|  90   |  global   | Transformer_E  |   set4   | longitude | -23hr~0hr |   1hr    |    0.581    |   12, 128   | min-max, no  |
|  91   |  global   | Transformer_E  |   set5   | longitude | -23hr~0hr |   1hr    |    0.687    |   12, 128   | min-max, no  |
|  92   |  global   | Transformer_E  |   set6   | longitude | -23hr~0hr |   1hr    |    0.730    |   12, 128   | min-max, no  |
|  93   |  global   | Transformer_E  |   set4   | latitude  | -3hr~0hr  |   1hr    |    0.504    |   12, 128   |     None     |
|  94   |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    0.561    |   12, 128   | min-max, no  |
|  95   |  global   | Transformer_E  |   set3   | longitude | -23hr~0hr |   6hr    |    1.581    |   12, 128   | min-max, no  |
|  96   |  global   | Transformer_E  |   set4   | longitude | -23hr~0hr |   6hr    |    1.580    |   12, 128   | min-max, no  |
|  97   |  global   | Transformer_E  |   set5   | longitude | -23hr~0hr |   6hr    |    1.879    |   12, 128   | min-max, no  |
|  98   |  global   | Transformer_E  |   set6   | longitude | -23hr~0hr |   6hr    |    2.000    |   12, 128   | min-max, no  |
|       |           |                |          |           |           |          |             |             |              |
|  100  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    1.045    |    6, 32    | min-max, no  |
|  101  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    0.656    |   12, 32    | min-max, no  |
|  102  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    0.673    |   18, 32    | min-max, no  |
|  103  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    1.032    |   24, 32    | min-max, no  |
|  104  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    0.620    |    6, 64    | min-max, no  |
|  105  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    0.586    |   12, 64    | min-max, no  |
|  106  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    0.596    |   18, 64    | min-max, no  |
|  107  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    0.599    |   24, 64    | min-max, no  |
|  108  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    0.573    |   18, 128   | min-max, no  |
|  109  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    2.360    |   24, 128   | min-max, no  |
|  110  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    0.560    |   6, 256    | min-max, no  |
|  111  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    1.244    |   12, 256   | min-max, no  |
|  112  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    3.847    |   18, 256   | min-max, no  |
|  113  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    0.560    |   24, 256   | min-max, no  |
|  114  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    0.889    |   24, 128   | min-max, no  |
|  115  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   1hr    |    0.568    |   24, 128   | min-max, no  |
|       |           |                |          |           |           |          |             |             |              |
|  116  |  global   | Transformer_E  |   set7   | longitude | -23hr~0hr |   1hr    |    0.552    |   12, 128   | min-max, no  |
|  117  |  global   | Transformer_E  |   set8   | longitude | -23hr~0hr |   1hr    |    1.168    |   12, 128   | min-max, no  |
|  118  |  global   | Transformer_E  |  *set7   | longitude | -23hr~0hr |   1hr    |    0.560    |   12, 128   | min-max, no  |
|  119  |  global   | Transformer_E  |  *set3   | longitude | -23hr~0hr |   1hr    |    0.560    |   12, 128   | min-max, no  |
|  120  |  global   | Transformer_E  |  *set8   | longitude | -23hr~0hr |   1hr    |    1.004    |   12, 128   | min-max, no  |
|  121  |  global   | Transformer_E  |  *set4   | longitude | -23hr~0hr |   1hr    |    0.562    |   12, 128   | min-max, no  |
|  122  |  global   | Transformer_E  |  *set5   | longitude | -23hr~0hr |   1hr    |    0.671    |   12, 128   | min-max, no  |
|  123  |  global   | Transformer_E  |  *set6   | longitude | -23hr~0hr |   1hr    |    1.060    |   12, 128   | min-max, no  |
|       |           |                |          |           |           |          |             |             |              |
|  125  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   72hr   |    2.099    |   12, 128   | min-max, no  |
|  126  |  global   | Transformer_E  |   set1   | longitude | -23hr~0hr |   96hr   |    2.201    |   12, 128   | min-max, no  |
|       |           |                |          |           |           |          |             |             |              |
|  130  |           |                |          |           |           |          |             |             |              |
|       |           |                |          |           |           |          |             |             |              |
|  135  |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   72hr   |    2.035    |   12, 128   | min-max, no  |
|  136  |  global   | Transformer_E  |   set1   |   time    | -23hr~0hr |   96hr   |    2.120    |   12, 128   | min-max, no  |
|  137  |  global   | Transformer_E  |   set1   | latitude  | -23hr~0hr |   72hr   |    2.228    |   12, 128   | min-max, no  |
|  138  |  global   | Transformer_E  |   set1   | latitude  | -23hr~0hr |   96hr   |    2.336    |   12, 128   | min-max, no  |
|       |           |                |          |           |           |          |             |             |              |
|  140  |  global   | Transformer_E  |  *set2   | longitude | -23hr~0hr |   1hr    |    0.648    |   12, 128   | min-max, no  |
|  141  |  global   | Transformer_E  |  *set9   | longitude | -23hr~0hr |   1hr    |    0.570    |   12, 128   | min-max, no  |
|  142  |  global   | Transformer_E  |  *set10  | longitude | -23hr~0hr |   1hr    |    0.895    |   12, 128   | min-max, no  |
|  143  |  global   | Transformer_E  |  *set11  | longitude | -23hr~0hr |   1hr    |    0.566    |   12, 128   | min-max, no  |
|  144  |  global   | Transformer_E  |   set2   | longitude | -23hr~0hr |   1hr    |    0.648    |   12, 128   | min-max, no  |
|  145  |  global   | Transformer_E  |   set9   | longitude | -23hr~0hr |   1hr    |    0.562    |   12, 128   | min-max, no  |
|  146  |  global   | Transformer_E  |  set10   | longitude | -23hr~0hr |   1hr    |    0.557    |   12, 128   | min-max, no  |
|  147  |  global   | Transformer_E  |  set11   | longitude | -23hr~0hr |   1hr    |    0.566    |   12, 128   | min-max, no  |
|       |           |                |          |           |           |          |             |             |              |
|  150  |  global   | Transformer_E  |   set7   | longitude | -23hr~0hr |   1hr    |    0.560    |   12, 128   | min-max, no  |
|  151  |  global   | Transformer_E  |  *set10  | longitude | -23hr~0hr |   1hr    |    0.564    |   12, 128   | min-max, no  |

- features set:  
    - set1: TEC  
    - set2: TEC, SW  
    - set3: TEC, DOY, hour  
    - set4: TEC, DOY, hour, long  
    - set5: TEC, DOY, hour, long, SW  
    - set6: TEC, DOY, hour, long, SW, label  
    - set7: TEC, long  
    - set8: TEC, DOY, SW, label  
    - set9: TEC, DOY  
    - set10: TEC, hour  
    - set11: TEC, label  
    - *setn: DOY, hour and long use min-max  

time (24, 71*72+2) (DOY, hour, SW)
lattitude (71, 24*72+3) (GTEC, DOY, hour, latitude)
longitude (72, 24*71)

1. 輸入時間維度
2. 輸出時間維度
3. input feature (date, sin cos method)