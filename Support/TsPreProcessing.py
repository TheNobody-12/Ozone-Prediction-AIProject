import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from numpy import array
def pre_processing(data):
    data['From Date'] = data['From Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    data['To Date'] = data['To Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    data['From Date'].min(), data['From Date'].max()

    aq_df = data.set_index('To Date')
    aq_df.drop(['From Date'], axis=1, inplace=True)
    aq_df.head()
    return aq_df

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X),array(y)

def Transform_Normalize(data):
    norm = data
    norm_arr = np.asarray(norm)
    norm = np.reshape(norm_arr, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    norm = scaler.fit_transform(norm)
    for i in range(5):
        print(norm[i])
    count = 0
    for i in range(len(norm)):
        if norm[i] == 0:
            count = count + 1 
    print('Number of null values in norm = ', count)
    #removing null values 
    norm = norm[norm!=0]
    return norm

