import pandas as pd

metrics = ["acc", "gmean", "kappa", "gmean_affected"]

for metric in metrics:

    global_acc = pd.read_csv("./semi-synth/summarized_{}.csv".format(metric), index_col=["id", "n_class"])
    drift_period_acc = pd.read_csv("./semi-synth/summarized_drifted_{}.csv".format(metric), index_col=["id", "n_class"])
    col_l1 = ["Global"]*14
    global_acc.columns = pd.MultiIndex.from_arrays([global_acc.columns, col_l1],names=['L0','L1'])
    col_l2 = ["Local Drift"]*14
    drift_period_acc.columns = pd.MultiIndex.from_arrays([drift_period_acc.columns, col_l2],names=['L0','L1'])
    #print (global_acc)
    #print (drift_period_acc)

    complete_df = pd.concat([global_acc, drift_period_acc], axis=1)
    complete_df = complete_df.reindex(sorted(complete_df.columns), axis=1)
    complete_df.reset_index(inplace=True)
    complete_df.to_csv("./semi-synth/{}_paper.csv".format(metric), index=True)
    print (complete_df)
