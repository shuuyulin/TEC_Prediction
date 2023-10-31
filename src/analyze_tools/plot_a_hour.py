import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import argparse
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
from PIL import Image
from src.importer import initialize_import

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='prediction.csv')
    parser.add_argument('-t', '--time', type=str, default='2021.11.4.12') # 2021/11/4 12hr
    parser.add_argument('-r', '--record', type=str, default='./')
    parser.add_argument('-tf', '--truth_path', type=str, default='./data/SWGIM3.0_year')
    parser.add_argument('-y', '--year', type=str, nargs=2, default=[2020, 2021])
    
    return parser

def read_truth_data(DATAPATH, years=[2020,2021]):
    
    print('Reading csv data...')
    
    # global
    df_list = []
    use_columns = list(range(3)) + list(range(10, 5122))
    for year in tqdm(years):
        year_df = pd.read_csv(DATAPATH / Path(f'{year}.csv'),\
            header=list(range(6)), index_col=0)
        
        year_df = year_df.iloc[:, use_columns]
        
        df_list.append(year_df)
        
    truth_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    
    return truth_df

def get_alpha_blend_cmap(cmap, alpha):
    cls = plt.get_cmap(cmap)(np.linspace(0,1,256))
    cls = (1-alpha) + alpha*cls
    return ListedColormap(cls)

def plot_heatmap_on_earth_pic(truth_np, pred_np, IMAGEPATH, RECORDPATH): # plot castleline with wrong picture
        
    # 載入背景圖片
    background_image = Image.open(IMAGEPATH).resize((710,720))

    truth_np = truth_np.reshape((71, 72))
    pred_np = pred_np.reshape((71, 72))

    # 設置畫布大小
    fig, axes = plt.subplots(
        ncols=3,
        sharex=True,
        # sharey=True,
        figsize=(10, 4),
        # gridspec_kw=dict(width_ratios=[4,4,4,0.4,0.4]),
        )
    cbar_ax1 = fig.add_axes([.9, .3, .02, .4])
    cbar_ax2 = fig.add_axes([.95, .3, .02, .4])
    xtick = [f'{i}' for i in np.arange(-180, 180, 5)]
    ytick = [f'{i}' for i in np.arange(87.5, -88, -2.5)]
    # 繪製heatmap
    for idx, (ax, data) in enumerate(zip(axes[:3], [truth_np, pred_np, abs(truth_np-pred_np)])):
        sns.heatmap(data,
                    xticklabels=xtick,
                    yticklabels=ytick if idx == 0 else False,
                    square=True,
                    cmap='jet' if idx != 2 else 'Greens',
                    # cmap=get_alpha_blend_cmap('jet' if idx != 2 else 'Greens', 0.5 ),
                    annot=False,
                    alpha=0.5,
                    linewidths=0.0,
                    edgecolor="none",
                    cbar=idx in (0, 2),
                    vmin=0,
                    vmax=40 if idx != 2 else 5,
                    cbar_ax = cbar_ax1 if idx == 0 else cbar_ax2 if idx == 2 else None,
                    ax=ax)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 10)
        # ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 10)
        if ax.is_last_row():
            ax.set_xlabel('latitude')
        if ax.is_first_col():
            ax.set_ylabel('longtitude')
        print('lens:',len(ax.get_xticklabels()), len(ax.get_yticklabels()))
        for index, xlabel in enumerate(ax.get_xticklabels()):
            k = 10
            vis = index % k == 0
            xlabel.set_visible(vis)
        for index, ylabel in enumerate(ax.get_yticklabels()):
            k = 10
            vis = index % k == 0
            ylabel.set_visible(vis)

        # 將背景圖片顯示在畫布上
        ax.imshow(background_image,
                aspect = ax.get_aspect(),
                extent = ax.get_xlim() + ax.get_ylim(),
                #   zorder = 1, #put the map under the heatmap
                )

        # 設置x, y軸標籤
    # fig.subplots_adjust(bottom=0, right=0.9, top=1)
    axes[0].set(title='Truth')
    axes[1].set(title='Prediction')
    axes[2].set(title='Difference')
    fig.canvas.draw()
    fig.tight_layout(rect=[0, 0, .9, 1])
    # fig.colorbar(axes[0].collections[0], cax=axes[3])
    # fig.colorbar(axes[2].collections[0], cax=axes[4])
    # 儲存圖片
    fig.suptitle('2021/11/5 12:00 GTEC MAP', fontsize=16)
    plt.savefig(RECORDPATH / 'prediction_truth_diff.jpg', dpi=2000)

def plot_heatmap_on_earth_car(truth_np, pred_np, IMAGEPATH, RECORDPATH):  # plot castleline with cartopy
        
    extent = (-180, 175, -87.5, 87.5)
    # 設置畫布大小
    fig, axes = plt.subplots(
        ncols=3,
        sharex=True,
        # sharey=True,
        figsize=(15, 4),
        # gridspec_kw=dict(width_ratios=[4,4,4,0.4,0.4]),
        subplot_kw={'projection': ccrs.PlateCarree()},
        )
    cbar_ax1 = fig.add_axes([.9, .3, .02, .4])
    cbar_ax2 = fig.add_axes([.95, .3, .02, .4])
    # 繪製heatmap
    for idx, (ax, data) in enumerate(zip(axes[:3], [truth_np, pred_np, abs(truth_np-pred_np)])):
        data = data.reshape((71, 72))
        # lat = [f'{i:.1f}' for i in np.linspace(extent[2],extent[3],data.shape[0])]
        # lon = [f'{i:.1f}' for i in np.linspace(extent[0],extent[1],data.shape[1])]
        lat = np.linspace(extent[3],extent[2],data.shape[0])
        lon = np.linspace(extent[0],extent[1],data.shape[1])
        # print(lon, lat)
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.3, color='black')
        gl = ax.gridlines(draw_labels=True, color = "None", crs=ccrs.PlateCarree(),)
        gl.xlabel_style = dict(fontsize=6)
        gl.ylabel_style = dict(fontsize=6)
        gl.top_labels = False
        gl.right_labels = False
        gl.right_labels = True if idx == 0 else False

        Lat,Lon = np.meshgrid(lat,lon)
        # if idx == 0:
        #     print(Lon, Lat)
        ax.pcolormesh(Lon,Lat,
                      np.transpose(data),
                      vmin=0,
                      vmax=40 if idx != 2 else 3,
                      cmap='jet' if idx != 2 else 'Greens',
                    #   cbar_ax = cbar_ax1 if idx == 0 else cbar_ax2 if idx == 2 else None,
                    #   cbar=idx in (0, 2),
                     )
        
        # ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 10)
        # ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 10)
        for index, xlabel in enumerate(ax.get_xticklabels()):
            k = 10
            vis = index % k == 0
            xlabel.set_visible(vis)
        for index, ylabel in enumerate(ax.get_yticklabels()):
            k = 10
            vis = index % k == 0
            ylabel.set_visible(vis)

        if ax.is_last_row():
            ax.set_xlabel('latitude')
        if ax.is_first_col():
            ax.set_ylabel('longtitude')
        print('lens:',len(ax.get_xticklabels()), len(ax.get_yticklabels()))

    # fig.subplots_adjust(bottom=0, right=0.9, top=1)
    axes[0].set(title='Truth')
    axes[1].set(title='Prediction')
    axes[2].set(title='Difference')
    # fig.canvas.draw()
    fig.tight_layout(rect=[0, 0, .9, 1])
    # fig.tight_layout()
    fig.colorbar(axes[0].collections[0], cax=cbar_ax1)
    fig.colorbar(axes[2].collections[0], cax=cbar_ax2)
    # 儲存圖片
    fig.suptitle('2021/11/5 12:00 GTEC MAP', fontsize=16)
    plt.savefig(RECORDPATH / 'prediction_truth_diff.jpg', dpi=1000)

def plot_map(RECORDPATH, map, title='GTEC Map',figname='heatmap.jpg', vmax=None, vmin=0): # n plot castleline
    # print(error_dict.values())
    plt.clf()
    # print(map)
    map = map.reshape((71,72))
    ax = sns.heatmap(map, xticklabels=np.arange(-180, 180, 5), yticklabels=np.arange(87.5, -88, -2.5), square=True, vmin=vmin, vmax=vmax)
    print(map.shape)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 10)
    len_t = len(ax.get_xticklabels())
    for index, (xlabel, ylabel) in enumerate(zip(ax.get_xticklabels(), ax.get_yticklabels())):
        k = 5
        vis = False if index > len_t - k else (index % k == 0)
        xlabel.set_visible(vis)
        ylabel.set_visible(vis)
        
    ax.set(title=title, xlabel='latitude', ylabel='longtitude')
    plt.savefig(RECORDPATH / figname)

def datetime2idx(date):
    delta = date - datetime(2021, 1, 1, 0)
    return int(delta / timedelta(hours=1))

def main():

    parser = get_parser()
    args = parser.parse_args()
    
    BASEPATH = Path(__file__).parent
    TRUTHPATH = args.truth_path
    
    RECORDPATH = Path(args.record)
    # read truth dataframe
    # truth_df = read_truth_data(TRUTHPATH)
    truth_df = initialize_import(config={'data':{'test_year':', '.join([str(i) for i in args.year])}},
                                 mode='test',
                                 path=TRUTHPATH).import_data()
    
    use_columns = list(range(3)) + list(range(10, 5122))
    truth_df = truth_df.iloc[:, use_columns]
    
    pred_df = pd.read_csv(args.file, header=list(range(6)), index_col=0).reset_index(drop=True)
    # print(pred_df.info())
    # print(pred_df.head())
    date = datetime.strptime(args.time, '%Y.%m.%d.%H')
    # print(type(date.hour))
    # print(pred_df.iloc[2] == date.hour)

    truth_sr = truth_df.iloc[datetime2idx(date),3:].reset_index(drop=True)
    pred_sr = pred_df.iloc[datetime2idx(date),10:].reset_index(drop=True)


    
    # plot_map(RECORDPATH, truth_sr.to_numpy(), 'GTEC CODE Map', 'truth.jpg', vmax=40)
    # plot_map(RECORDPATH, pred_sr.to_numpy(), 'GTEC Transformer Map', 'prediction.jpg', vmax=40)
    # plot_map(RECORDPATH, truth_sr.sub(pred_sr,fill_value=0).to_numpy(), 'RMSE Map', 'error.jpg')#, vmin=-5, vmax=5)

    plot_heatmap_on_earth_car(truth_sr.to_numpy(), pred_sr.to_numpy(), './worldmap3.png', RECORDPATH)

if __name__ == "__main__":
    main()