import numpy as np
import pandas as pd
import os
os.chdir("/pyws") # working directory
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
from scipy.signal import periodogram


def generate_sequences(d, seqlen):
    X = []
    y = []
    for i in range(len(d)-2*seqlen):
        X.append(d[i:(i+seqlen)])
        y.append(d[(i+seqlen):(i+2*seqlen)])
    X, y = np.array(X), np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y.reshape((y.shape[0], y.shape[1], 1))

def standardize_and_split(d, train_ratio = .7, meta_train_ratio = .7, seqlen = 20, reverse = False):
    if reverse:
        d = list(reversed(d))
    train_lim = int(train_ratio*len(d))
    meta_train_lim = int(train_lim+(len(d)-train_lim)*meta_train_ratio)
    d_mean = d[:train_lim].mean()
    d_sd = d[:train_lim].std()
    d = (d-d_mean)/d_sd
    
    X_train, y_train = generate_sequences(d[:train_lim], seqlen)
    X_meta_train, y_meta_train = generate_sequences(d[train_lim:meta_train_lim], seqlen)
    X_test, y_test = generate_sequences(d[meta_train_lim:], seqlen)
    
    print("# seqs in X_train (y_train):", len(X_train), "(", len(y_train), ")")
    print("# seqs in X_meta_train (y_meta_train):", len(X_meta_train), "(", len(y_meta_train), ")")
    print("# seqs in X_test (y_test):", len(X_test), "(", len(y_test), ")")
    
    return X_train, y_train, X_meta_train, y_meta_train, X_test, y_test, d_mean, d_sd


def get_periodicity_fourier(series, k=5, seqlen_min=1, seqlen_max=1000):
    perio = periodogram(series, fs=1)
    perio_df = pd.DataFrame({"freq": perio[0], "spec": perio[1]})
    perio_df.sort_values(by="spec", ascending=False, inplace=True) # sort decreasingly
    top_freqs = perio_df["freq"].values
    top_lags = np.array([round(1/x) for x in top_freqs if x!=0])
    # unique lags only
    _, idx = np.unique(top_lags, return_index=True)
    top_k_lags = top_lags[np.sort(idx)]
    top_k_lags = np.int_(top_k_lags)
    top_k_lags = top_k_lags# [:k]
    top_k_lags = top_k_lags[top_k_lags>=seqlen_min]
    top_k_lags = top_k_lags[top_k_lags<=seqlen_max]
    top_k_lags = top_k_lags[:k]
    if len(top_k_lags) < k:
        print("Warning: Too little Fourier seqlens available.")
    return top_k_lags



### By default, a new snapshot is trained based on random init weights
### To specify a pretrained model, set model_init_seqlen accordingly
def train_snapshot_lstm(d, d_name, seqlens = np.arange(10, 101, 10), num_fc = None, epochs_per_seqlen=10, batch_size=50,
                       model_init_seqlen = None, no_of_returned_best_seqlens = None, dir_periodicity=None):
    if num_fc is None:
        num_fc = min(seqlens)
    if model_init_seqlen is None:
        model = Sequential()
        # Input Layer
        model.add(LSTM(input_shape=(None, 1), units=64, return_sequences=True))
        model.add(Dropout(.2))
        # Hidden Layer
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(.2))
        # output layer
        model.add(TimeDistributed(Dense(units=1)))
        model.add(Activation("linear"))
        model.compile(loss="mse", optimizer="adam")
        model.summary()
        
    else:
        # Load most recent model state and continue training
        model = load_model('models/'+d_name+'_snapshots/Dec17/'+dir_periodicity+'/'+'snap_seqlen_'+str(model_init_seqlen)+'.h5')
    rmses = {}
    for seqlen in seqlens:
        print("seqlen: "+str(seqlen)+"=====")
        X_train, y_train, X_meta_train, y_meta_train, X_test, y_test, d_mean, d_sd = standardize_and_split(d, seqlen=seqlen)
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs_per_seqlen, validation_split=0.1, 
                  verbose=False)
        model.save('models/'+d_name+'/snapshots/snapshot_'+str(seqlen)+'.h5')
        # Evaluate on test data
        yhat_meta_train = model.predict(X_meta_train)[:, :num_fc]
        curr_rmse = np.sqrt((((yhat_meta_train*d_sd+d_mean) - (y_meta_train[:, :num_fc]*d_sd+d_mean)) ** 2).mean())
        print("RMSE of Snap", seqlen, ":", curr_rmse)
        rmses["snap"+str(seqlen)] = curr_rmse
    return rmses


### Training function for single optimized LSTM
def train_lstm_full(d, d_name, seqlen, num_fc = None, epochs=100, batch_size=50):
    model = Sequential()
    # Input Layer
    model.add(LSTM(input_shape=(None, 1), units=64, return_sequences=True))
    model.add(Dropout(.2))
    # Hidden Layer
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(.2))
    # output layer
    model.add(TimeDistributed(Dense(units=1)))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    X_train, y_train, X_meta_train, y_meta_train, X_test, y_test, d_mean, d_sd = standardize_and_split(d, seqlen=seqlen)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)
    # store model
    model.save('models/'+d_name+'/singleopt/00_singleOpt_seqlen_'+str(seqlen)+'.h5')

    # Evaluate on test data
    yhat_meta_train = model.predict(X_meta_train)[:, :num_fc]
    curr_rmse = np.sqrt((((yhat_meta_train*d_sd+d_mean) - (y_meta_train[:, :num_fc]*d_sd+d_mean)) ** 2).mean())
    print("RMSE", seqlen, ":", curr_rmse)
    
def train_lstm_base_classic(d, d_name, seqlen, num_fc = None, epochs=100, batch_size=50):
    model = Sequential()
    # Input Layer
    model.add(LSTM(input_shape=(None, 1), units=64, return_sequences=True))
    model.add(Dropout(.2))
    # Hidden Layer
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(.2))
    # output layer
    model.add(TimeDistributed(Dense(units=1)))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    X_train, y_train, X_meta_train, y_meta_train, X_test, y_test, d_mean, d_sd = standardize_and_split(d, seqlen=seqlen)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1)
    # store model
    model.save('models/'+d_name+'/baseclassic/baseclassic_seqlen_'+str(seqlen)+'.h5')

    # Evaluate on test data
    yhat_meta_train = model.predict(X_meta_train)[:, :num_fc]
    curr_rmse = np.sqrt((((yhat_meta_train*d_sd+d_mean) - (y_meta_train[:, :num_fc]*d_sd+d_mean)) ** 2).mean())
    print("RMSE", seqlen, ":", curr_rmse)



#### Example Usage ####
### Train Full LSTM with single best seqlen over 100 epochs (snapshots will have 5 epochs for each of the 20 snapshots)
d = pd.read_csv("data/river_flows.csv", sep = ";")
d = np.array(d["flow"].values)
seqlens_fourier_riverflow = get_periodicity_fourier(d, k=20, seqlen_min=50, seqlen_max=100)[0]
train_lstm_full(d, "riverflow", seqlen = seqlens_fourier_riverflow, num_fc=10)

### Train 20 Snapshots a 5 epochs
d = pd.read_csv("data/river_flows.csv", sep = ";")
d = np.array(d["flow"].values)
seqlens_fourier_riverflow = get_periodicity_fourier(d, k=20, seqlen_min=50, seqlen_max=100)
train_snapshot_lstm(d, "riverflow", seqlens = seqlens_fourier_riverflow, num_fc = 10, epochs_per_seqlen=5, batch_size=50,
                       model_init_seqlen = None, no_of_returned_best_seqlens = None)

### Train classic base models, 5 epochs each
d = pd.read_csv("data/river_flows.csv", sep = ";")
d = np.array(d["flow"].values)
seqlens_fourier_riverflow = get_periodicity_fourier(d, k=20, seqlen_min=50, seqlen_max=100)
for seqlen in seqlens_fourier_riverflow:
    train_lstm_base_classic(d, "riverflow", seqlen = seqlen, num_fc=10, epochs=5)