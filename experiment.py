import os
import multiprocessing
import random
import xml.etree.ElementTree as ET
import warnings
from scipy.spatial.distance import cdist
import sys

from mapie.metrics import regression_coverage_score
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_percentage_error
from splc2py.fmodel import FeatureModel
from splc2py.sampling import Sampler
import pandas as pd
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
import json

import conf_pred
import oracleFunction
from sklearn.utils import shuffle

'''
STRATEGIES = {
    "twise_1_25_400": {
        "train": {"binary": "twise", "numeric": False,
                  "params": {"t": 1}},
        "calib": 25,
        "test": 400},
    "twise_2_25_400": {
        "train": {"binary": "twise", "numeric": False,
                  "params": {"t": 2}},
        "calib": 25,
        "test": 400},
    "twise_3_25_400": {
        "train": {"binary": "twise", "numeric": False,
                  "params": {"t": 3}},
        "calib": 25,
        "test": 400},
    "random_15_25_400": {"train": {"binary": "random", "params": {"numConfigs": 15, "sampleSize": 15}},
                         "calib": 25,
                         "test": 400},
    "random_101_25_400": {"train": {"binary": "random", "params": {"numConfigs": 101, "sampleSize": 101}},
                          "calib": 25,
                          "test": 400},
    "random_407_25_400": {"train": {"binary": "random", "params": {"numConfigs": 407, "sampleSize": 407}},
                          "calib": 25,
                          "test": 400},
    "random_40_0_400": {"train": {"binary": "random", "params": {"numConfigs": 40, "sampleSize": 40}},
                        "calib": 0,
                        "test": 400},
    "random_126_0_400": {"train": {"binary": "random", "params": {"numConfigs": 126, "sampleSize": 126}},
                         "calib": 0,
                         "test": 400},
    "random_432_0_400": {"train": {"binary": "random", "params": {"numConfigs": 432, "sampleSize": 432}},
                         "calib": 0,
                         "test": 400}

}


METHODS = [
    ['cv_plus', 'lin', None],
    ['jackknife_plus', 'lin', None],

    ['cv_plus', 'lin_lasso', None],
    ['jackknife_plus', 'lin_lasso', None],

    ['cv_plus', 'cart', None],
    ['jackknife_plus', 'cart', None],

    ['cv_plus', 'rf', None],
    ['jackknife_plus', 'rf', None],

    ['cv_plus', 'kr', None],
    ['jackknife_plus', 'kr', None],

    # ['cv_plus', 'svr', None],
    # ['jackknife_plus', 'svr', None],

    ['cqr', 'lin_quant', None],
    ['cqr', 'lgbm_quant', None],

]
'''

def find_closest_row(sample_row, measurements, numeric_features):
    sample_numeric = sample_row[numeric_features].values.reshape(1, -1)
    measurements_numeric = measurements[numeric_features].values
    distances = cdist(sample_numeric, measurements_numeric, 'euclidean')
    closest_index = np.argmin(distances)
    return measurements.iloc[closest_index]


def read_measurements(samples, path_to_measurements, binary_features, numeric_features, nfp):
    measurements = pd.read_csv(path_to_measurements, sep=';')
    if numeric_features:
        scaler = preprocessing.StandardScaler()
        scaler.fit(measurements[numeric_features])
        samples[numeric_features] = scaler.transform(samples[numeric_features])
        measurements[numeric_features] = scaler.transform(measurements[numeric_features])
    for index, sample_row in samples.iterrows():
        # Find corresponding row in measurements
        matching_row = measurements[
            (measurements[binary_features] == sample_row[binary_features]).all(axis=1)
        ]
        closest_row = find_closest_row(sample_row, matching_row, numeric_features)
        # Extract nfp value and add it to the sample row
        nfp_value = closest_row[nfp]
        samples.at[index, nfp] = nfp_value
        # Update numeric features in the sample row
        samples.loc[index, numeric_features] = closest_row[numeric_features].values
    if numeric_features:
        samples[numeric_features] = scaler.inverse_transform(samples[numeric_features])
    return samples


# Define your sampling function here
def sampling(folder, config, random_state, strategy):
    path_to_measurements = os.path.abspath(os.path.join(folder, os.pardir, os.pardir, "measurements.csv"))
    vm_path = os.path.abspath(os.path.join(folder, os.pardir, os.pardir, "FeatureModel.xml"))

    strategies = config["strategies"]
    nfp = config["nfp"]

    # print(strategies)
    vm = ET.parse(vm_path)
    fm = FeatureModel(vm)

    sampler = Sampler(vm, "docker")

    columns = []
    columns.extend(fm.binary)
    columns.extend(fm.numeric)
    columns.append(nfp)

    measurements_all = pd.read_csv(path_to_measurements, sep=';')
    measurements = measurements_all[columns]

    # Train-Samples
    seed = random.randint(1, 10000)
    params = strategies[strategy]["train"]
    print(params["binary"])
    params["params"].update(seed=seed)
    if params["binary"] == "random":
        # params["params"].update(seed=seed)
        # samples = sampler.sample(binary=params["binary"], params=params["params"], formatting="dict")
        # if len(fm.numeric) > 0:
        #     params["params"].update(numConfigs=1)
        # samples_num = sampler.sample(binary=params["binary"], numeric="random", params=params["params"],
        #                              formatting="dict")
        # train_samples = pd.DataFrame.from_dict(samples)
        # train_samples_num = pd.DataFrame.from_dict(samples_num)
        # train_samples = pd.concat([train_samples, train_samples_num[fm.numeric]], axis=1).reindex(train_samples.index)
        train_samples = measurements.sample(n=(params["params"]["numConfigs"]), random_state=seed, ignore_index=True)
    elif params["binary"] == "distance-based":
        print("debug")
        samples = sampler.sample(hybrid="distribution-aware", params=params["params"],
                                 formatting="dict")
        train_samples = pd.DataFrame.from_dict(samples)
        train_samples = read_measurements(train_samples, path_to_measurements, fm.binary, fm.numeric, nfp)
        train_samples = train_samples.drop_duplicates(keep='first')
    elif params["binary"] == "twise":
        t = params["params"]["t"]
        for i in range(t):
            params["params"]["t"] = i + 1
            if params["numeric"]:
                samples = sampler.sample(binary=params["binary"], numeric=params["numeric"], params=params["params"],
                                         formatting="dict")
            else:
                samples = sampler.sample(binary=params["binary"], params=params["params"],
                                         formatting="dict")
            if i == 0:
                train_samples = pd.DataFrame.from_dict(samples)
            else:
                train_samples = train_samples.append(samples, ignore_index=True)

        train_samples = read_measurements(train_samples, path_to_measurements, fm.binary, fm.numeric, nfp)
        train_samples = train_samples.drop_duplicates(keep='first')
    else:
        if params["numeric"]:
            samples = sampler.sample(binary=params["binary"], numeric=params["numeric"], params=params["params"],
                                     formatting="dict")
        else:
            samples = sampler.sample(binary=params["binary"], params=params["params"],
                                     formatting="dict")
        train_samples = pd.DataFrame.from_dict(samples)
        train_samples = read_measurements(train_samples, path_to_measurements, fm.binary, fm.numeric, nfp)
        train_samples = train_samples.drop_duplicates(keep='first')

    # Calib-Samples
    num_calib_samples = strategies[strategy]["calib"]
    if num_calib_samples > 0:
        seed = random.randint(1, 10000)
        calib_samples = measurements.sample(n=(num_calib_samples + train_samples.shape[0]), random_state=seed,
                                            ignore_index=True)
        calib_samples = pd.concat([train_samples, calib_samples], ignore_index=True, sort=False)
        calib_samples = calib_samples.drop_duplicates(keep='first')
        calib_samples = calib_samples.iloc[train_samples.shape[0]:]
        calib_samples = calib_samples.iloc[calib_samples.shape[0] - num_calib_samples:]
        seed = random.randint(1, 10000)
        num_test_samples = strategies[strategy]["test"]
        test_samples = measurements.sample(n=(num_test_samples + train_samples.shape[0] + calib_samples.shape[0]),
                                           random_state=seed, ignore_index=True)
        test_samples = pd.concat([train_samples, calib_samples, test_samples], ignore_index=True, sort=False)
        test_samples = test_samples.drop_duplicates(keep='first')
        test_samples = test_samples.iloc[(train_samples.shape[0] + num_calib_samples):]
        test_samples = test_samples.iloc[(test_samples.shape[0] - num_test_samples):]
    else:
        seed = random.randint(1, 10000)
        num_test_samples = strategies[strategy]["test"]
        test_samples = measurements.sample(n=(num_test_samples + train_samples.shape[0]),
                                           random_state=seed, ignore_index=True)
        test_samples = pd.concat([train_samples, test_samples], ignore_index=True, sort=False)
        test_samples = test_samples.drop_duplicates(keep='first')
        test_samples = test_samples.iloc[(train_samples.shape[0]):]
        test_samples = test_samples.iloc[(test_samples.shape[0] - num_test_samples):]
        calib_samples = pd.DataFrame()
    '''
    seed = random.randint(1, 10000)
    num_calib_samples = STRATEGIES[strategy]["calib"]
    samples = sampler.sample(binary="random",
                             params={"seed": seed, "numConfigs": (num_calib_samples + train_samples.shape[0])},
                             formatting="dict")
    samples_num = sampler.sample(binary="random", numeric="random",
                                 params={"seed": seed, "sampleSize": (
                                         num_calib_samples + train_samples.shape[0]),
                                         "numConfigs": 1},
                                 formatting="dict")


    calib_samples = pd.DataFrame.from_dict(samples)
    calib_samples_num = pd.DataFrame.from_dict(samples_num)

    calib_samples = pd.concat([calib_samples, calib_samples_num[fm.numeric]], axis=1).reindex(calib_samples.index)

    calib_samples = read_measurements(calib_samples, path_to_measurements, fm.binary, fm.numeric, NFP)

    calib_samples = pd.concat([train_samples, calib_samples], ignore_index=True, sort=False)
    calib_samples = calib_samples.drop_duplicates(keep='first')
    calib_samples = calib_samples.iloc[train_samples.shape[0]:]
    calib_samples = calib_samples.iloc[calib_samples.shape[0] - num_calib_samples:]
    '''

    # Test-Samples
    '''
    seed = random.randint(1, 10000)
    num_test_samples = STRATEGIES[strategy]["test"]
    samples = sampler.sample(binary="random",
                             params={"seed": seed,
                                     "numConfigs": (
                                             num_test_samples + train_samples.shape[0] + calib_samples.shape[0])},
                             formatting="dict")
    samples_num = sampler.sample(binary="random", numeric="random",
                                                        params={"seed": seed, "sampleSize": (
                                                                num_test_samples + train_samples.shape[0] +
                                                                calib_samples.shape[0]), "numConfigs": 1},
                                                        formatting="dict")

    test_samples = pd.DataFrame.from_dict(samples)
    test_samples_num = pd.DataFrame.from_dict(samples_num)
    test_samples = pd.concat([test_samples, test_samples_num[fm.numeric]], axis=1).reindex(test_samples.index)

    print(test_samples.shape)
    test_samples = read_measurements(test_samples, path_to_measurements, fm.binary, fm.numeric, NFP)
    print(test_samples.shape)
    test_samples = pd.concat([train_samples, calib_samples, test_samples], ignore_index=True, sort=False)
    test_samples = test_samples.drop_duplicates(keep='first')
    test_samples = test_samples.iloc[(train_samples.shape[0] + num_calib_samples):]
    test_samples = test_samples.iloc[(test_samples.shape[0] - num_test_samples):]
    print(test_samples.shape)
    '''
    # Give Values to Samples
    # Toy Systems

    # Toy Systems:
    '''
    oracle = oracleFunction.Oraclefunction()
    oracle.set_features(feature_list)
    influences_path = os.path.abspath(os.path.join(folder, os.pardir, os.pardir, "influences.csv"))
    oracle.set_influences(pd.read_csv(influences_path).to_dict())
    influences = oracle.get_influences()

    for index, row in train_samples.iterrows():
        train_samples.at[index, 'Energie'] = oracle.get_value(row.to_dict())

    for index, row in calib_samples.iterrows():
        calib_samples.at[index, 'Energie'] = oracle.get_value(row.to_dict())

    for index, row in test_samples.iterrows():
        test_samples.at[index, 'Energie'] = oracle.get_value(row.to_dict())

    '''

    train_samples = pd.DataFrame.reset_index(train_samples, drop=True)
    calib_samples = pd.DataFrame.reset_index(calib_samples, drop=True)
    test_samples = pd.DataFrame.reset_index(test_samples, drop=True)

    print(train_samples.shape)
    print(calib_samples.shape)
    print(test_samples.shape)

    return train_samples, calib_samples, test_samples, fm


def get_yerr(y_pred, y_pis_min, y_pis_max):
    return np.concatenate(
        [
            abs(np.expand_dims(y_pred, 0) - y_pis_min),
            abs(y_pis_max - np.expand_dims(y_pred, 0))
        ],
        axis=0,
    )


def learning(folder, config, random_state, strategy, train, calib, test, fm):
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

    for method in config["METHODS"]:
        try:
            if calib.empty and method[0] == "cqr":
                continue
            warnings.filterwarnings("ignore")
            print(f"Start Method {method}")
            seed = random.randint(0, 100000)
            pred = conf_pred.cplearning(method=method[1], strategy=method[0], conformity_score=method[2], nfp=nfp,
                                       train_data=train, calib_data=calib, test_data=test, alpha=alpha,
                                       train_calib_split=0.6, random_state=seed)
            filename = f"{method[0]}_{method[1]}_{method[2]}_pred.csv"
            pred.to_csv(os.path.join(folder, filename), index=False)
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
            with open(os.path.join(folder, filename), "w") as fp:
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
            plt.savefig(os.path.join(folder, filename))

        except:
            print("" + method[0] + "," + method[1] + " failed!!")
            pass

    return 0


# Function to create subfolders and run sampling function in parallel
def create_folders_and_run(folder_path, config, n):
    strategies = config["strategies_names"]
    for strategy in strategies:
        strategy_folder = os.path.join(folder_path, strategy)
        os.makedirs(strategy_folder, exist_ok=True)
        # Run sampling function n times in parallel
        pool = multiprocessing.Pool()
        for i in range(1, n + 1):
            run_folder = os.path.join(strategy_folder, f"run_{i}")
            os.makedirs(run_folder, exist_ok=True)
            pool.apply_async(run_experiment, args=(run_folder, config, strategy,))
        pool.close()
        pool.join()
    return 0


# Function to run sampling and save output
def run_experiment(folder, config, strategy):
    random_state = random.getstate()
    train, calib, test, fm = sampling(folder, config, random_state, strategy)
    train.to_csv(os.path.join(folder, "train.csv"), index=False)
    calib.to_csv(os.path.join(folder, "calib.csv"), index=False)
    test.to_csv(os.path.join(folder, "test.csv"), index=False)
    with open(os.path.join(folder, 'output.txt'), 'w') as f:
        f.write(f"Random state: {random_state}\n")
    learning(folder, config, random_state, strategy, train, calib, test, fm)


# Example usage
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        folder_path = args[0]
    else:
        # folder_path = "/home/sjahns/Experimente/Experiment_x264"
        folder_path = "/mnt/e/Experiment_x264_energy"
    json_name = "config.json"
    json_path = os.path.join(folder_path, json_name)
    with open(json_path) as json_file:
        config = json.load(json_file)

    n = 30  # Number of runs for each strategy
    print(folder_path)
    create_folders_and_run(folder_path, config, n)
