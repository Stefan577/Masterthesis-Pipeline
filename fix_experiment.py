import random
import sys
import os
import json
import multiprocessing
from statistics import mean
import xml.etree.ElementTree as ET

from mapie.metrics import regression_coverage_score
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.utils import shuffle
import warnings

import numpy as np
import pandas as pd
from splc2py.fmodel import FeatureModel

import ConfPred
from experiment import get_yerr




def get_folder_names(path):
    folder_names = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path) and item not in ['$RECYCLE.BIN', 'System Volume Information']:
            folder_names.append(item)

    return folder_names


def rerun_experiment(path, config, rerun):
    print(f"Try to rerun: {rerun}\n")

    print("Load Dataframes\n")
    run_path = os.path.join(path, rerun[0], rerun[1])
    train = pd.read_csv(os.path.join(run_path, "train.csv"))
    print(train)
    calib = pd.read_csv(os.path.join(run_path, "calib.csv"))
    print(calib)
    test = pd.read_csv(os.path.join(run_path, "test.csv"))
    print(test)

    print("FeatureModel\n")
    vm_path = os.path.abspath(os.path.join(run_path, os.pardir, os.pardir, "FeatureModel.xml"))

    # print(strategies)
    vm = ET.parse(vm_path)
    fm = FeatureModel(vm)

    train = shuffle(train)
    if not calib.empty:
        calib = shuffle(calib)

    train = pd.DataFrame.reset_index(train, drop=True)
    if not calib.empty:
        calib = pd.DataFrame.reset_index(calib, drop=True)

    alpha = config["alpha"]
    nfp = config["nfp"]

    X_train = train.drop(columns=nfp)
    y_train = train.loc[:, nfp]

    X_test = test.drop(columns=nfp)
    y_test = test.loc[:, nfp]

    method = rerun[2]

    warnings.filterwarnings("ignore")
    print(f"Start Method {method}")
    seed = random.randint(0, 100000)
    pred = ConfPred.cplearning(method=method[1], strategy=method[0], conformity_score=method[2], nfp=nfp,
                               train_data=train, calib_data=calib, test_data=test, alpha=alpha,
                               train_calib_split=0.6, random_state=seed)
    filename = f"{method[0]}_{method[1]}_{method[2]}_pred.csv"
    pred.to_csv(os.path.join(run_path, filename), index=False)
    coverage = regression_coverage_score(
        pred["measured"].tolist(), pred["interval_min"].tolist(), pred["interval_max"].tolist()
    )

    y_err = get_yerr(pred["predicted"].to_numpy(), pred["interval_min"].to_numpy(),
                     pred["interval_max"].to_numpy())
    pred_int_width = (
            pred["interval_max"].to_numpy() - pred["interval_min"].to_numpy()
    )

    mean_int_width = (mean(pred_int_width))

    mape = mean_absolute_percentage_error(y_pred=pred["predicted"], y_true=y_test)

    df = pd.concat([X_test, pred], axis=1)
    df['interval_length'] = df['interval_max'] - df['interval_min']
    data_boxplot = []
    for f in fm.binary:
        rest_df = df.loc[df[f] == 1]
        data_boxplot.append(rest_df['interval_length'].to_numpy())

    y_pred = pred["predicted"].to_numpy()

    results = {'coverage': coverage, 'mean_int_length': mean_int_width, 'mape': mape, 'seed': seed}
    filename = f"{method[0]}_{method[1]}_{method[2]}_result.txt"
    with open(os.path.join(run_path, filename), "w") as fp:
        json.dump(results, fp)
    fig, axs = plt.subplots(3, 1, figsize=(5, 15))

    axs[0].errorbar(
        y_test,
        y_pred,
        yerr=y_err,
        alpha=0.5,
        linestyle="None",
    )
    axs[0].scatter(y_test, y_pred, s=1, color="black")
    axs[0].plot(
        [0, max(max(y_test), max(y_pred))],
        [0, max(max(y_test), max(y_pred))],
        "-r",
    )
    axs[0].set_xlabel("Actual Energy")
    axs[0].set_ylabel("Predicted Energy")
    axs[0].grid()
    axs[0].set_title(f"{method}\ncoverage={coverage:.0%}\nError:{mape:.3}")
    xmin, xmax = axs[0].get_xlim()

    axs[1].scatter(y_test, pred_int_width, marker="+")
    axs[1].set_xlabel("Actual Energy")
    axs[1].set_ylabel("Prediction interval width")
    axs[1].grid()
    axs[1].set_xlim([xmin, xmax])

    axs[1].set_title(f"mean intervall length:\n {mean_int_width:.5}")
    axs[2].boxplot(data_boxplot)
    axs[2].set_xlabel("Features")
    axs[2].set_ylabel("Prediction interval width")

    fig.suptitle(
        f"Predicted values with the prediction intervals of level {alpha}"
    )
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # plt.show()
    filename = f"{method[0]}_{method[1]}_{method[2]}_plot.png"
    plt.savefig(os.path.join(run_path, filename))


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        exp_path = args[0]
    else:
        # folder_path = "/home/sjahns/Experimente/Experiment_x264"
        exp_path = "/mnt/e/Experiment_nginx_pervolution_energy"
    json_name = "config.json"
    json_path = os.path.join(exp_path, json_name)
    with open(json_path) as json_file:
        config = json.load(json_file)
    methods = config["METHODS"]

    rerun_list = []

    sampling_strategies = get_folder_names(exp_path)
    sampling_strategies.remove("old")
    print(sampling_strategies)
    runs = np.empty(len(sampling_strategies), dtype=object)
    for i in range(len(sampling_strategies)):
        runs[i] = get_folder_names(os.path.join(exp_path, sampling_strategies[i]))
        print(runs[i])
    for m in range(len(methods)):
        ml_models = methods[m]
        for s in range(len(sampling_strategies)):
            for r in runs[s]:
                json_name = f"{ml_models[0]}_{ml_models[1]}_{ml_models[2]}_result.txt"
                json_path = os.path.join(exp_path, sampling_strategies[s], r, json_name)
                if not ("_0_" in sampling_strategies[s] and ml_models[0] == "cqr"):
                    try:
                        with open(json_path) as json_file:
                            data = json.load(json_file)
                    except FileNotFoundError:
                        rerun_list.append([sampling_strategies[s], r, ml_models])
                        print(f"{sampling_strategies[s]}: {r} has no Data for this ML-Model({ml_models})")

    print(rerun_list)

    # pool = multiprocessing.Pool()
    for rerun in rerun_list:
        rerun_experiment(exp_path, config, rerun)
