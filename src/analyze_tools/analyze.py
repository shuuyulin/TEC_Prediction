import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tqdm import tqdm
import scipy as sp
import json
from datetime import datetime, timedelta
from src.importer import initialize_import
# from GTEC_Prediction.src.importer
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='prediction.csv')
    parser.add_argument('-m', '--mode', type=str, default='global')
    parser.add_argument('-y', '--year', type=int, nargs=2, default=[2020, 2021])
    parser.add_argument('-r', '--record', type=str, default='./')
    parser.add_argument('-tf', '--truth_path', type=str, default='./data/SWGIM3.0_year')
    parser.add_argument('-o', '--output_file', type=str, default='record.json') #filename, store in record path
    # single / global
    
    return parser

def compare_prediction_head(mode, reserved, truth_df, pred_df, RECORDPATH):
    # prepare plot dataframe
    reserved = 5000
    head = reserved + 24*15
    
    col = pred_df.columns[-1]
    x_date = truth_df['UTC'].iloc[reserved:head].apply(lambda x:\
        datetime(x['year'], 1, 1, hour=x['hour']) + timedelta(days=x['DOY'][0].item())\
        , axis=1)
    # print(x_date)
    if mode == 'global':
        truth_df = truth_df[('CODE', 'GIM')].iloc[reserved:head].apply(
            (lambda x: x.mean()), axis=1)
        pred_df = pred_df[col[0]].iloc[reserved:head].apply(
            (lambda x: x.mean()), axis=1)
    else:
        truth_df = truth_df[('CODE', *col[1:])].iloc[reserved:head]
        pred_df = pred_df[col].iloc[reserved:head]
        
    x_date = x_date.reset_index(drop=True)
    truth_df = truth_df.reset_index(drop=True)
    pred_df = pred_df.reset_index(drop=True)
        
    # print(pred_df.head())
    # print(truth_df.head())
    
    plot_df = pd.concat([x_date, pred_df, truth_df], axis=1)
    plot_df.columns = ['date', 'prediction', 'truth']
    plot_df = plot_df.melt('date', var_name='cols', value_name='vals')
    
    plt.clf()
    plt.title(f'Comparison of Prediction and Truth')
    ax = sns.lineplot(x='date', y='vals', data=plot_df, hue='cols')
    ax.set(xlabel='Date', ylabel='TEC value (10TEC)')
    plt.xticks(rotation=40)
    plt.tight_layout()
    
    plt.legend()
    plt.savefig(RECORDPATH / Path(f'./compare_prediction_head.jpg'))
    # plt.savefig(RECORDPATH / Path(f'./compare_prediction_head_{int(reserved/24)}.jpg'))

def compare_prediction_daily(mode, truth_df, pred_df, RECORDPATH):
    # collecting data into a dataframe
    
    col = pred_df.columns
    pred_df = pred_df.groupby(col[1]).mean()[col[-1]]
    truth_df = truth_df.groupby(col[1]).mean()[('CODE', *col[-1][1:])]
    
    new_df = pd.DataFrame({'prediction':pred_df, 'truth':truth_df})
    # print(new_df.info())
    plt.clf()
    plt.title('Comparison of Prediction and Truth (daily average)')
    ax = sns.lineplot(data=new_df, legend='brief')
    ax.set(xlabel='DOY', ylabel='TEC (10TEC)')

    plt.savefig(RECORDPATH / Path(f'./compare_prediction_daily.jpg'))
    
def compare_prediction_hourly(mode, truth_df, pred_df, RECORDPATH):
    # collecting data into a dataframe
    
    col = pred_df.columns
    pred_df = pred_df.groupby(col[2]).mean()[col[-1]]
    truth_df = truth_df.groupby(col[2]).mean()[('CODE', *col[-1][1:])]
    
    new_df = pd.DataFrame({'prediction':pred_df, 'truth':truth_df})
    # print(new_df.info())
    plt.clf()
    plt.title('Comparison of Prediction and Truth (hourly average)')
    ax = sns.lineplot(data=new_df, marker="o", legend='brief')
    ax.set(xlabel='hour', ylabel='TEC (10TEC)')

    plt.savefig(RECORDPATH / Path(f'./compare_prediction_hourly.jpg'))

def plot_error_historgram(mode, pred_df, RECORDPATH):
    errors = pred_df[(pred_df.columns[-1][0],'PRED-OBS')].to_numpy().reshape((-1))
    plt.clf()
    max_val = np.nanmax([abs(error) for error in errors])
    errors = np.array(errors)
    errors = errors[~np.isnan(errors)]
    ax = sns.histplot(errors[~np.isnan(errors)], binwidth=5, binrange=(-max_val,max_val))
    ax.set(xlabel='error (10TEC)', ylabel='count')
    plt.title('Error Historgram')
    plt.savefig(RECORDPATH / Path('his_gram.png'))

def plot_correlation(mode, truth_df, pred_df, RECORDPATH):
    print(truth_df.info())
    print(pred_df.info())
    model_name = pred_df.columns[-1][0]
    truthes = truth_df[('CODE', 'GIM')].iloc[24+3:].to_numpy().reshape((-1))
    preds = pred_df[(model_name,'GIM')].iloc[24+3:].to_numpy().reshape((-1))
    
    plt.clf()
    scatter_size = 1 if mode == 'single' else 0.0002
    
    print('Regression ploting...')
    ax = sns.regplot(x=truthes, y=preds,\
        line_kws={"color": "red", 'linewidth':1}, scatter_kws={'s':scatter_size})
    ax.set(xlabel='CODE estimated  TEC value',\
        ylabel=f'{model_name} predicting  TEC value')
    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(\
        truthes, preds)
    
    ax.text(5, 550, f'$R^2\ :$ {r_value:.4f}\n$Cor:$ {slope:.4f}', fontsize=9) #add text
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    
    print(f'r^2={r_value:.4f}, cor={slope:.4g}')
    plt.title('Regression and Correlation Analyze')
    plt.grid()
    plt.savefig(RECORDPATH / Path('corr.png'))
    return r_value, slope
    
def plot_error_heatmap(error_data, title, RECORDPATH):
    
    # print(error_dict.values())
    plt.clf()
    error_map = np.array(error_data).reshape((71,72))
    # print(error_map)
    print('max:', error_map.max())
    extent = (-180, 175, -87.5, 87.5)
    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8), projection=ccrs.PlateCarree())
    cbar_ax1 = fig.add_axes([.9, .3, .02, .4])

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.3, color='black')
    gl = ax.gridlines(draw_labels=True, color = "None", crs=ccrs.PlateCarree(),)
    gl.xlabel_style = dict(fontsize=6)
    gl.ylabel_style = dict(fontsize=6)
    gl.top_labels = False
    gl.right_labels = False

    lat = np.linspace(extent[3],extent[2],error_map.shape[0])
    lon = np.linspace(extent[0],extent[1],error_map.shape[1])
    Lat,Lon = np.meshgrid(lat,lon)
    ax.pcolormesh(Lon,Lat,
                    np.transpose(error_map),
                    vmin=0,
                    vmax=2,
                    cmap='Greens',
                    )
        
    fig.colorbar(ax.collections[0], cax=cbar_ax1)
    fig.tight_layout(rect=[0, 0, .8, 1])
    ax.set(title=title)
    plt.savefig(RECORDPATH / f'{"_".join(title.split(" "))}.jpg', dpi=500)


def read_truth_data(DATAPATH, years=[2020,2021]):
    
    # print('Reading csv data...')
    # if Path(DATAPATH).is_file():
    #     truth_df = pd.read_csv(DATAPATH, header=list(range(6)), index_col=0)
    #     truth_df = truth_df.iloc[:,[0,1,2,8,9,10,11,12,13]]
        
    #     return truth_df
    
    # global
    df_list = []
    use_columns = list(range(3)) + list(range(10, 5122))
    for year in tqdm(years):
        year_df = pd.read_csv(DATAPATH / Path(f'{year}.csv'),\
            header=list(range(6)), index_col=0)
        
        year_df = year_df.iloc[:, use_columns]
        # rename dataframe
        # year_df.columns = renamelist
        
        df_list.append(year_df)
        
    truth_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    
    return truth_df

def cal_mse(errors):
    return np.nansum([error**2 for error in errors]) / np.count_nonzero(~np.isnan(errors))

def CheckLeap(Year):
    # Checking if the given year is leap year
    if((Year % 400 == 0) or
        (Year % 100 != 0) and
            (Year % 4 == 0)):
        return True
    # Else it is not a leap year
    else:
        return False
    
if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()
    
    BASEPATH = Path(__file__).parent
    TRUTHPATH = args.truth_path
    
    RECORDPATH = Path(args.record)
    # read truth dataframe
    truth_df = initialize_import(config={'data':{'test_year':', '.join([str(i) for i in args.year])}},mode='test', path=args.truth_path).import_data()
    # truth_df = read_truth_data(TRUTHPATH, args.year)

    use_columns = list(range(3)) + list(range(10, 5122))
    truth_df = truth_df.iloc[:, use_columns]
    
    # read prediction dataframe
    pred_df = pd.read_csv(args.file, header=list(range(6)), index_col=0).reset_index(drop=True)
                
    k = len(pred_df.columns) - 71*72
    use_cols = list(range(3)) + list(range(k, k+71*72))
    pred_df = pred_df.iloc[:, use_cols]
    
    # linear correlation
    r_value, cor = None, None
    # r_value, cor = plot_correlation(args.mode, truth_df, pred_df, RECORDPATH)
    
    RMSE_dict = {}
    year_map = np.zeros((len(args.year), 71*72))
    tt_MSE = 0
    # count error
    for idx, col in enumerate(pred_df.columns[3:]):
        # print(truth_df[('CODE','GIM','TECU','f2.1','85.0','-180')].info())
        errors = [t - p for t, p in zip(truth_df[('CODE', *col[1:])], pred_df[col])]
                
        # calculate yearly MSE
        spliter = 366*24 if CheckLeap(args.year[0]) else 365*24
        year_RMS = [cal_mse(errors[:spliter]), cal_mse(errors[spliter:])]
        
        for year_i in range(year_map.shape[0]):
            year_map[year_i,idx] = year_RMS[year_i]

        RMSE_dict[f'({col[4]}, {col[5]})'] = np.power(sum(year_RMS), 0.5)
        tt_MSE += sum(year_RMS)
    
    # error historgram
    # plot_error_historgram(args.mode, pred_df, RECORDPATH)

    RMSE_overall = np.power(tt_MSE / len(pred_df.columns[3:]), 0.5)
    print(f'overall_RMSE: {RMSE_overall}')
    
    # save RMSE & R^2 & correlation
    json.dump({'r_value':r_value, 'correlation':cor, 'RMSE_overall':RMSE_overall,'RMSE':RMSE_dict},\
        open(RECORDPATH / args.output_file, 'w'), indent=4)
    
    # for i, year in enumerate(args.year):
    #     plot_error_heatmap(year_map[i], f'{year} RMSE Global MAP', RECORDPATH)

    # plot_error_heatmap(list(RMSE_dict.values()), f'RMSE Global MAP', RECORDPATH)
    
    # save error csv file
    # pred_df.to_csv(RECORDPATH / Path('./prediction_error.csv'))
    
