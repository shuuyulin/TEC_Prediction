import statistics
import pandas as pd
class Norm():
    def __init__(self):
        pass
    def fit(self, data):
        return 0
    def normalize(self, data, *args): # data only one dim
        return data
    def denormalize(self, data, *args):
        return data
    
class MinMaxNorm(Norm):
    def __init__(self):
        pass
    def fit(self, data):
        return min(data), max(data)
    def normalize(self, data, _min, _max): # data only one dim
        return (data - _min) / (_max - _min)
    def denormalize(self, data, _min, _max):
        return data * (_max - _min) + _min

class StandardNorm(Norm):
    def __init__(self):
        pass
    def fit(self, data):
        return statistics.mean(data), statistics.variance(data) # mean and variance
    def normalize(self, data, _mean, _var): # data only one dim
        return (data - _mean) / _var
    def denormalize(self, data, _mean, _var):
        return data * _var + _mean
    
class HibertNorm(Norm):
    def __init__(self):
        super().__init__()
    def fit(self, data):
        return 0
    def normalize(self, data, limit): # data only one dim
        return pd.concat((pd.sin(data / limit * 2 * pd.pi), pd.cos(data / limit * 2 * pd.pi)), dim=-1)
    def denormalize(self, data, *args):
        return data
    
    