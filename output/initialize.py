from .functions import *

def exporting(config, *args, **kwargs):
    output_func_list = {
        'SWGIM': SWGIM_export,
    }

    function_name = config['output']['output_func']
    rounding_digit = int(config['output']['rounding_digit'])
    
    if function_name in output_func_list:
        return output_func_list[function_name](config, rounding_digit, *args, **kwargs)
    else:
        print(f'There is no output function called {function_name}.')
        raise AttributeError