from ..utils import *
from .initialize import initialize_norm
def preprocess_data(config, df):
    # drop features and normalize
    slicing = {
        'year' : (0, 1),
        'DOY' : (1, 2),
        'hour' : (2, 3),
        'kp' : (3, 4),
        'r' : (4, 5),
        'dst' : (5, 6),
        'ap' : (6, 7),
        'f10.7' : (7, 8),
        'storm_state' : (8, 9),
        'storm_size' : (9, 10),
        'tec' : (10, 10+71*72),
        'tec_er' : (10+71*72, 10+71*72*2),
        'tec_sh' : (10+71*72*2, 10+71*72*2+256),
        'tec_sh_er' : (10+71*72*2+256, 10+71*72*2+256*2),
    }
    input_features = config2strlist(config['data']['input_features'])
    truth_features = config2strlist(config['data']['truth_features'])
    input_norm_type = config2strlist(config['preprocess']['input_norm_type'])
    truth_norm_type = config2strlist(config['preprocess']['truth_norm_type'])
    # print(input_norm_type)
    
    data = {'input':{}, 'truth':{}}
    for ft_col, norm_type in zip(input_features, input_norm_type):
        ft = df.iloc[:,slice(*slicing[ft_col])].to_numpy()
        # normalize
        if norm_type != "None":
            norm, norm_params = initialize_norm(norm_type)
            if ft_col not in ['tec', 'tec_er', 'tec_sh', 'tec_sh_er']:
                ft = norm.normalize(ft, *norm_params[ft_col])
            else:
                for i in range(ft.shape[1]):
                    ft[:,i] = norm.normalize(ft[:,i], *norm_params[ft_col][i])
        data['input'][ft_col] = ft
        logging.debug(f'{ft_col}: {ft.shape}')
        # print('input')
        # print(ft_col, np.min(ft), np.max(ft), ft.shape)
        
    for ft_col, norm_type in zip(truth_features, truth_norm_type):
        ft = df.iloc[:,slice(*slicing[ft_col])].to_numpy()
        # print(ft_col, ft.shape, ft, (np.min(ft), np.max(ft)))
        # normalize
        if norm_type != "None":
            norm, norm_params = initialize_norm(norm_type)
            if ft_col not in ['tec', 'tec_er', 'tec_sh', 'tec_sh_er']:
                ft = norm.normalize(ft, *norm_params[ft_col])
            else:
                for i in range(ft.shape[1]):
                    ft[:,i] = norm.normalize(ft[:,i], *norm_params[ft_col][i])
        # print(ft_col, ft.shape, ft, (np.min(ft), np.max(ft)))
        data['truth'][ft_col] = ft
        # print('truth')
        # print(ft_col, np.min(ft), np.max(ft), ft.shape)
        
    return data