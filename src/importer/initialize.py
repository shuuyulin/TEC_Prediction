from .importers import *
from pathlib import Path
from src.utils import *
def initialize_import(config, mode, *args, **kwargs):
    
    # path: args[0]
    type_map = {
        'csv' : csv_importer,
        'h5' : hdf_importer,
    }
    
    file_type = next(Path(kwargs['path']).iterdir()).suffix[1:]
    
    years = config2intlist(config['data'][f'{mode}_year'])
    
    return type_map[file_type](kwargs['path'], years)
    
    