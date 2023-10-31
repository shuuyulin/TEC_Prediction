import pandas as pd
from tqdm import tqdm
from pathlib import Path
import logging

class importer():
    def __init__(self, path:Path, years) -> None:
        self.path = path
        self.years = years
    def import_data(self):
        df = pd.DataFrame()
        return df

class csv_importer(importer):
    def __init__(self, path:Path, years) -> None:
        self.path = path
        self.years = years
        
    def import_data(self):

        df_list = []
        logging.info('Reading csv data...')
        for year in tqdm(self.years, dynamic_ncols=True):
            year_df = pd.read_csv(self.path / Path(f'{year}.csv'),\
                header=list(range(6)), index_col=0)
                        
            df_list.append(year_df)
            
        all_df = pd.concat(df_list, axis=0).reset_index(drop=True)
                    
        return all_df
        
class hdf_importer(importer):
    def __init__(self, path:Path, years) -> None:
        self.path = path
        self.years = years
    def import_data(self):
        df_list = []
        logging.info('Reading csv data...')
        for year in tqdm(self.years, dynamic_ncols=True):
            year_df = pd.read_hdf(self.path / Path(f'{year}.h5'), key='data')
                        
            df_list.append(year_df)
            
        all_df = pd.concat(df_list, axis=0).reset_index(drop=True)
                    
        return all_df