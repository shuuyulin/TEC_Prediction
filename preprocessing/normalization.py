from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ==TODO==
# X 輸入標準化 輸出是值
# V 輸入標準化 輸出標準化
# 
def initialize_normalization(config):
    
    normalization_type_list = {
        'max_min' : MinMaxScaler,
        'z_score' : StandardScaler,
    }
    
    normalization_type = config['data']['normalization_type']
    
    if normalization_type in normalization_type_list:
        norm = normalization_type_list[normalization_type]()
    else:
        print('normalization_type has not been defined in config file!')
        raise AttributeError
    return norm
