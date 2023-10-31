from .seqbase import *
def initialize_shaper(config, *args, **kwargs):
    type_map = {
        'time' : time_shaper,
        'latitude' : lat_shaper,
        'longitude' : long_shaper,
    }
    
    return type_map[config['data']['seq_base']](config)