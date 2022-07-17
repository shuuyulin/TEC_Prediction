import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
point = '(25, 120)'
def compare_prediction(df_truth, df_pred):
    plt.clf()
    plt.title('compare predict truth')
    plt.plot(df_truth[point], label='truth')
    plt.plot(df_pred[point], label='pred')
    
    plt.legend()
    # plt.show()
    plt.savefig('compare_prediction.jpg')
    
if __name__ == '__main__':
    BASEPATH = Path(__file__).parent
    df_truth = pd.read_csv(BASEPATH / Path('../data/single_point_test.csv'))
    df_pred = pd.read_csv(BASEPATH / Path('../record/2/prediction.csv'))
    compare_prediction(df_truth.iloc[:365], df_pred.iloc[:365])
    