import pandas as pd
import numpy as np

# rc = pd.DataFrame([],columns=[1,2,3], index=list(range(6)))
# a = np.array([[1,2,3],[4,5,6]])
# print(a)
# a = np.concatenate([[[None]*a.shape[1]]*4, a],axis=0)
# print(a)
# print(rc)
# rc.iloc[:] = a
# rc = [[None]*4]*5

# print(rc)
# print(f'{12345.6789:7.7f}')
# pred = np.load(open('predict.npy', 'rb'), allow_pickle=True)
# print(pred[0])
# print(np.shape(pred))
# Python program explaining
# round_() function
import numpy as np

in_array = [.5, 1.5, 2.5, 3.5, 4.5, 0.6]
print ("Input array : \n", in_array)

round_off_values = np.round(in_array)
print ("\nRounded values : \n", round_off_values)


in_array = [.53, 1.54, .71]
print ("\nInput array : \n", in_array)

round_off_values = np.round_(in_array)
print ("\nRounded values : \n", round_off_values)

in_array = [.5538, 1.33354, .71445]
print ("\nInput array : \n", in_array)

round_off_values = np.round_(in_array, decimals = 1)
print ("\nRounded values : \n", round_off_values)
