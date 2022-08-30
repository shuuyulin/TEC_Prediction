import statistics
# ==TODO==
# X 輸入標準化 輸出是值
# V 輸入標準化 輸出標準化
# 
class MinMaxNorm():
    def __init__(self):
        pass
    def fit(self, data):
        return min(data), max(data)
    def normalize(self, data, _min, _max): # data only one dim
        return (data - _min) / (_max - _min)
    def denormalize(self, data, _min, _max):
        return data * (_max - _min) + _min

class StandardNorm():
    def __init__(self):
        pass
    def fit(self, data):
        return statistics.mean(data), statistics.variance(data) # mean and variance
    def normalize(self, data, _mean, _var): # data only one dim
        return (data - _mean) / _var
    def denormalize(self, data, _mean, _var):
        return data * _var + _mean
    