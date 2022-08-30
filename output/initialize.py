from .functions import *

def initialize_output_func(config, *args, **kwargs):
    output_func_list = {
        'SWGIM': SWGIM_outpu_func,
    }

    function_name = config['output']['output_func']
    
    if function_name in output_func_list:
        return output_func_list[function_name](*args, **kwargs)
    else:
        print(f'There is no model called {function_name}.')
        raise AttributeError