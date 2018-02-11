import numpy as np
import pandas as pd
import os
os.chdir("/pyws") # working directory
from keras.models import load_model
from scipy.signal import periodogram
from sklearn.linear_model import Ridge
from collections import OrderedDict


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

#### Evaluate snapshot ensembles and compare to single, optimized model
def meta_forecast(meta_train_forecasts, meta_train_actual, meta_test_forecasts, meta_test_actual, num_trees = 100,
                  ridge_alphas = [20, 15, 10, 8, 6, 4, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 
                                  0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001],
                  meta_learner = "rf"):
    meta_train_X = None
    meta_train_y = []
    meta_test_X = None
    meta_test_y = []

    for model in meta_train_forecasts:
        model_seqs = []
        for seq in model:
            model_seqs.extend(seq)
        if meta_train_X is None:
            meta_train_X = model_seqs
        else:
            meta_train_X = np.vstack((meta_train_X, model_seqs))

    meta_train_X = meta_train_X.transpose()    

    for seq_actual in meta_train_actual:
        meta_train_y.extend(seq_actual)
    
    # generate test data
    for model in meta_test_forecasts:
        model_seqs = []
        for seq in model:
            model_seqs.extend(seq)
        if meta_test_X is None:
            meta_test_X = model_seqs
        else:
            meta_test_X = np.vstack((meta_test_X, model_seqs))
    meta_test_X = meta_test_X.transpose()  
    
    for seq_actual in meta_test_actual:
        meta_test_y.extend(seq_actual)
        
    # Learn meta model    
    if meta_learner is "ridge":
        print("Now learning ridge regressor with alphas:", ridge_alphas, "...")
        best_ridge_rmse = np.Inf
        best_alpha = None
        best_forecasts = None
        for ridge_alpha in ridge_alphas:
            meta_learner = Ridge(alpha=ridge_alpha, normalize=True)
            meta_learner.fit(meta_train_X, meta_train_y)
            meta_forecasts = meta_learner.predict(meta_test_X)
            # evaluate
            meta_rmse = rmse(np.array(meta_forecasts)*data_sd+data_mean, np.array(meta_test_y)*data_sd+data_mean)
            if meta_rmse < best_ridge_rmse:
                best_ridge_rmse = meta_rmse
                best_alpha = ridge_alpha
                best_forecasts = meta_forecasts
        print("Best Ridge result with alpha="+str(best_alpha)+": RMSE="+str(best_ridge_rmse))
        return best_forecasts, best_ridge_rmse

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def fc_lstm_ensemble(d, model_filename_base, num_fc, method = "snapshot"):
    global data_mean, data_sd
    rmse_all = OrderedDict()
    if method is "snapshot":
        # get model names sorted by creation date (oldest model -> best seqlen)
        search_dir = "models/"+model_filename_base+"/snapshots"
        models_names = [s for s in os.listdir(search_dir) if os.path.isfile(os.path.join(search_dir, s))]
        models_names.sort(key=lambda s: os.path.getmtime(os.path.join(search_dir, s)))
        models_seqlens = [m.split("_")[1] for m in models_names]
        models_seqlens = np.array([m.split(".")[0] for m in models_seqlens], dtype=np.int)
    if method is "classic":
        search_dir = "models/"+model_filename_base+"/baseclassic"
        models_names = [s for s in os.listdir(search_dir) if os.path.isfile(os.path.join(search_dir, s))]
        models_seqlens = [m.split("_")[2] for m in models_names]
        models_seqlens = np.array([m.split(".")[0] for m in models_seqlens], dtype=np.int)
        
    offset = max(models_seqlens)+num_fc
    
    train_ratio = .7
    train_lim = int(train_ratio*len(d))
    data_mean = d[:train_lim].mean()
    data_sd = d[:train_lim].std()
    d = (d-data_mean)/data_sd
    holdoutset = d[train_lim:]
    upper_limit = round(0.8*len(holdoutset))
    
    meta_train_input = []
    meta_train_forecasts = []
    meta_train_actual = []
    meta_test_forecasts = []
    meta_test_actual = []
    
    for current_h5_name in models_names:
        if method is "snapshot":
            current_seq_length = current_h5_name.split("_")[1]
        if method is "classic":
            current_seq_length = current_h5_name.split("_")[2]
        current_seq_length = np.int(current_seq_length.split(".")[0])
        print("current_h5_name:", current_h5_name, ", current_seq_length:", current_seq_length)
        if method is "snapshot":
            model = load_model('models/'+model_filename_base+"/snapshots/"+current_h5_name)
        if method is "classic":
            model = load_model('models/'+model_filename_base+"/baseclassic/"+current_h5_name)
        meta_train = holdoutset[:upper_limit]
        meta_test = holdoutset[upper_limit:]
        # generate training data for stacking model
        meta_train_seqs = []
        for index in range(len(meta_train)-offset):
            meta_train_seqs.append(meta_train[index:index+offset])
        meta_train_seqs = np.array(meta_train_seqs)
        meta_train_seqs_X = meta_train_seqs[:, len(meta_train_seqs[0])-num_fc-current_seq_length:len(meta_train_seqs[0])-num_fc]
        meta_train_seqs_y = meta_train_seqs[:, -num_fc:]
        meta_train_input.append(meta_train_seqs_X[:, -num_fc:]) # additional features for meta-learner. num_fc = min(sequence_lengths)!

        meta_train_seqs_X = np.reshape(meta_train_seqs_X, (meta_train_seqs_X.shape[0], meta_train_seqs_X.shape[1], 1))
        meta_train_seqs_y = np.reshape(meta_train_seqs_y, (meta_train_seqs_y.shape[0], meta_train_seqs_y.shape[1], 1))
        meta_train_seqs_forecasts = model.predict(meta_train_seqs_X)[:, :num_fc]

        meta_train_forecasts.append(np.reshape(meta_train_seqs_forecasts, (len(meta_train_seqs_forecasts), len(meta_train_seqs_forecasts[0]))))
        meta_train_actual = np.reshape(meta_train_seqs_y, (len(meta_train_seqs_y), len(meta_train_seqs_y[0]))) # is identical for each seqlen 


        # compute model forecasts and errors on the actual test data
        meta_test_seqs = []
        for index in range(len(meta_test)-offset):
            meta_test_seqs.append(meta_test[index:index+offset])
        meta_test_seqs = np.array(meta_test_seqs)
        print("Evaluating on", len(meta_test_seqs), "test sequences.")
        meta_test_seqs_X = meta_test_seqs[:, len(meta_test_seqs[0])-num_fc-current_seq_length:len(meta_test_seqs[0])-num_fc]
        meta_test_seqs_y = meta_test_seqs[:, -num_fc:]

        meta_test_seqs_X = np.reshape(meta_test_seqs_X, (meta_test_seqs_X.shape[0], meta_test_seqs_X.shape[1], 1))
        meta_test_seqs_y = np.reshape(meta_test_seqs_y, (meta_test_seqs_y.shape[0], meta_test_seqs_y.shape[1], 1))
        
        meta_test_seqs_forecasts = model.predict(meta_test_seqs_X)[:, :num_fc]
        meta_test_forecasts.append(np.reshape(meta_test_seqs_forecasts, (len(meta_test_seqs_forecasts), len(meta_test_seqs_forecasts[0]))))
        meta_test_actual = np.reshape(meta_test_seqs_y, (len(meta_test_seqs_y), len(meta_test_seqs_y[0]))) # is identical for each seqlen 
        curr_rmse = rmse(meta_test_seqs_forecasts*data_sd+data_mean, meta_test_seqs_y*data_sd+data_mean)

        rmse_all["Seqlen"+str(current_seq_length)] = curr_rmse
        print(current_h5_name+" RMSE:", curr_rmse)
        
    
    meta_test_forecasts = np.array(meta_test_forecasts)# dim: (# models, # test sequences, # forecasts)
    meta_train_forecasts = np.array(meta_train_forecasts)
    meta_train_actual = np.array(meta_train_actual)
    meta_test_forecasts = np.array(meta_test_forecasts)
    
    # Evaluate ensembles of varying size
    print("meta_test_forecasts.shape:", meta_test_forecasts.shape)
    print("meta_train_forecasts.shape:", meta_train_forecasts.shape)
    if method is "snapshot":
        ensemble_range = np.arange(2, meta_test_forecasts.shape[0])
    if method is "classic":
        ensemble_range = [np.max(np.arange(2, meta_test_forecasts.shape[0]))]
    for ensemble_size in ensemble_range:
        # Actuals are independent of ensemble size
        current_meta_test_forecasts = meta_test_forecasts[:ensemble_size:]
        current_meta_train_forecasts = meta_train_forecasts[:ensemble_size:]
        
    
        print("current_meta_test_forecasts.shape:", current_meta_test_forecasts.shape)
        print("current_meta_train_forecasts.shape:", current_meta_train_forecasts.shape)
    
        # Mean Forecast
        # Important: Assuming len(rmse_all) == Number of individual base learners!
        meta_test_forecast_mean = np.mean(current_meta_test_forecasts, axis = 0)
        fc_mean_rmse = rmse(meta_test_forecast_mean*data_sd+data_mean, 
                                         meta_test_actual*data_sd+data_mean)
        num_baseLearners = len(rmse_all)
        rmses_baseLearners = rmse_all.copy()
        rmse_sum = 0
        for r in rmses_baseLearners.values():
            rmse_sum+=r
        print("Baselearner avg. RMSE:", round(rmse_sum/len(rmses_baseLearners), 2))
        percent_better_than_mean = len({k: v for k, v in rmses_baseLearners.items() if v < fc_mean_rmse}.values())/num_baseLearners
        rmse_all["MeanFC_"+str(ensemble_size)] = fc_mean_rmse
        print("Mean Forecast RMSE:", fc_mean_rmse, ", which is better than", str(round((1-percent_better_than_mean)*100, 2)), "% of the base learners.")

        # Ridge Regression Forecast
        fc_ridge = meta_forecast(current_meta_train_forecasts, meta_train_actual, current_meta_test_forecasts, meta_test_actual, 
                                 meta_learner="ridge")
        percent_better_than_ridge = len({k: v for k, v in rmses_baseLearners.items() if v < fc_ridge[1]}.values())/num_baseLearners
        rmse_all["RidgeFC_"+str(ensemble_size)] = fc_ridge[1]
        print("RidgeReg RMSE:", fc_ridge[1], ", which is better than", str(round((1-percent_better_than_ridge)*100, 2)), "% of the base learners.")
    
    # evaluate single optimized model
    search_dir = "models/"+model_filename_base+"/singlemodel"
    filelist = os.listdir(search_dir)
    filelist.sort()
    current_h5_name = filelist[0]
    current_seq_length = current_h5_name.split("_")[3]
    current_seq_length = np.int(current_seq_length.split(".")[0])
    

    print("current_h5_name:", current_h5_name, ", current_seq_length:", current_seq_length)
    model = load_model('models/'+model_filename_base+"/singlemodel/"+current_h5_name)
    meta_train = holdoutset[:upper_limit]
    meta_test = holdoutset[upper_limit:]
    # generate training data for stacking model
    meta_train_seqs = []
    for index in range(len(meta_train)-offset):
        meta_train_seqs.append(meta_train[index:index+offset])
    meta_train_seqs = np.array(meta_train_seqs)
    meta_train_seqs_X = meta_train_seqs[:, len(meta_train_seqs[0])-num_fc-current_seq_length:len(meta_train_seqs[0])-num_fc]
    meta_train_seqs_y = meta_train_seqs[:, -num_fc:]
    meta_train_input.append(meta_train_seqs_X[:, -num_fc:]) # additional features for meta-learner. num_fc = min(sequence_lengths)!
    meta_train_seqs_X = np.reshape(meta_train_seqs_X, (meta_train_seqs_X.shape[0], meta_train_seqs_X.shape[1], 1))
    meta_train_seqs_y = np.reshape(meta_train_seqs_y, (meta_train_seqs_y.shape[0], meta_train_seqs_y.shape[1], 1))
    meta_train_seqs_forecasts = model.predict(meta_train_seqs_X)[:, :num_fc]
    meta_train_actual = np.reshape(meta_train_seqs_y, (len(meta_train_seqs_y), len(meta_train_seqs_y[0]))) # is identical for each seqlen 


    # compute model forecasts and errors on the actual test data
    meta_test_seqs = []
    for index in range(len(meta_test)-offset):
        meta_test_seqs.append(meta_test[index:index+offset])
    meta_test_seqs = np.array(meta_test_seqs)
    print("Evaluating on", len(meta_test_seqs), "test sequences.")
    meta_test_seqs_X = meta_test_seqs[:, len(meta_test_seqs[0])-num_fc-current_seq_length:len(meta_test_seqs[0])-num_fc]
    meta_test_seqs_y = meta_test_seqs[:, -num_fc:]

    meta_test_seqs_X = np.reshape(meta_test_seqs_X, (meta_test_seqs_X.shape[0], meta_test_seqs_X.shape[1], 1))
    meta_test_seqs_y = np.reshape(meta_test_seqs_y, (meta_test_seqs_y.shape[0], meta_test_seqs_y.shape[1], 1))

    meta_test_seqs_forecasts = model.predict(meta_test_seqs_X)[:, :num_fc]
    meta_test_actual = np.reshape(meta_test_seqs_y, (len(meta_test_seqs_y), len(meta_test_seqs_y[0]))) # is identical for each seqlen 
    curr_rmse = rmse(meta_test_seqs_forecasts*data_sd+data_mean, meta_test_seqs_y*data_sd+data_mean)
   
    rmse_all["Singlemodel"] = curr_rmse
    print(current_h5_name+" RMSE:", curr_rmse)


#### Example Usage ####
d = pd.read_csv("data/births.csv", sep = ";")
d = np.array(d["no_births"].values)
fc_lstm_ensemble(d, "births", num_fc=10, method="snapshot")

d = pd.read_csv("data/births.csv", sep = ";")
d = np.array(d["no_births"].values)
fc_lstm_ensemble(d, "births", num_fc=10, method="classic")

