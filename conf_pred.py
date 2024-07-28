from rich.logging import RichHandler
import logging
import pandas as pd
import mlflow.sklearn
import mlflow
import time
from joblib import parallel_backend
from scipy.stats import randint, uniform

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.linear_model import LinearRegression, QuantileRegressor, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor

from mapie.conformity_scores import GammaConformityScore, AbsoluteConformityScore
from mapie.regression import MapieQuantileRegressor, MapieRegressor
from mapie.subsample import Subsample


def activate_logging(logs_to_artifact):
    with open("logs.txt", "w"):
        pass
    if logs_to_artifact:
        return logging.basicConfig(
            filename="logs.txt",
            level=logging.INFO,
            format="LEARNING    %(message)s",
        )
    return logging.basicConfig(
        level=logging.INFO,
        format="LEARNING    %(message)s",
        handlers=[RichHandler()],
    )


estimators = {
    "svr": SVR,
    "cart": DecisionTreeRegressor,
    "rf": RandomForestRegressor,
    "knn": KNeighborsRegressor,
    "kr": KernelRidge,
    "bagging": BaggingRegressor
}

tuning_params = {
    "svr": {
        "kernel": ["linear", "poly", "rbf"],
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 500, 750, 250, 1000.0],
        "gamma": [0.001, 0.01, 0.1, 0.005, 0.05, 0.5, 0.25, 0.025, 0.075, 1.0],
        "epsilon": [0.001, 0.01, 0.1, 0.5, 1.0],
    },
    "cart": {
        "min_samples_split": [],
        "min_samples_leaf": [],
        "ccp_alpha": [0.000001, 0.00001, 0.0001, 0.001, 0.01],
        "random_state": [1],
    },
    "rf": {
        "max_features": [],
        "n_estimators": list(range(2, 30)),
    },
    "knn": {
        "n_neighbors": list(range(2, 21)),
        "weights": ["distance", "uniform"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "p": list(range(1, 6)),
    },
    "kr": {
        "kernel": ["poly", "rbf", "linear"],
        "alpha": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.05],
        "degree": list(range(1, 6)),
        "gamma": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.05],
    },
    "bagging": {
        "base_estimator": [
            DecisionTreeRegressor(min_samples_leaf=i) for i in range(1, 11)
        ],
        "n_estimators": list(range(2, 21)),
    },
    "lin": {
        "poly__degree": [1]
    },
    "lin_lasso": {
        "poly__degree": [1, 2, 3, 4],
        "lasso__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]
    },
    "lin_quant": {
        "poly__degree": list(range(1, 4))
    },
    "lgbm_quant": {
        "num_leaves": [10, 20, 30, 40, 50],
        "max_depth": [-1, 3, 5, 10],
        "n_estimators": [2, 3, 4, 5, 10, 20],
        "learning_rate": [0.05, 0.1, 0.15, 0.2, 0.5, 0.9],
        "verbose": [-1]
    }
}


def _make_pred_artifact(pred, test_y: pd.Series):
    prediction = pd.concat([pd.Series(pred[0]), pd.Series(pred[1][:, 0, 0].T), pd.Series(pred[1][:, 1, 0].T), test_y],
                           axis=1)
    prediction.columns = ["predicted", "interval_min", "interval_max", "measured"]
    return prediction


def create_param_grid(method, n_features):
    param_space = tuning_params[method]

    if method == "cart":
        param_space["min_samples_split"] = list(range(2, n_features))
        param_space["min_samples_leaf"] = [
            round(1 / 3 * minsplit) for minsplit in param_space["min_samples_split"]
        ]
    elif method == "rf":
        param_space["max_features"] = list(range(2, n_features))

    return param_space


STRATEGIES = {
    "jackknife": {"method": "base", "cv": -1},
    "jackknife_plus": {"method": "plus", "cv": -1},
    "jackknife_minmax": {"method": "minmax", "cv": -1},
    "cv_plus": {"method": "plus", "cv": 10},
    "jackknife_plus_ab": {"method": "plus", "cv": Subsample(n_resamplings=50)},
    "cqr": {"method": "quantile", "cv": "split"},
}


def cplearning(
        method: str = "cart",
        strategy: str = "cv_plus",
        conformity_score=None,
        nfp: str = "",
        train_data=None,
        test_data=None,
        logs_to_artifact: bool = False,
        random_state: int = 42,
        alpha: float = 0.05,
        calib_data=pd.DataFrame(),
        train_calib_split=0.5
):
    activate_logging(logs_to_artifact)
    logging.info("Start learning from sampled configurations.")

    train_x = train_data.drop(columns=nfp)
    train_y = train_data.loc[:, nfp]
    test_x = test_data.drop(columns=nfp)
    test_y = test_data.loc[:, nfp]

    if strategy == "cqr":
        if calib_data.empty:
            train_x, calib_x, train_y, calib_y = train_test_split(
                train_x,
                train_y,
                random_state=random_state,
                test_size=train_calib_split
            )
        else:
            calib_x = calib_data.drop(columns=nfp)
            calib_y = calib_data.loc[:, nfp]
        if method == "lin_quant":
            model = Pipeline([
                ("poly", PolynomialFeatures(degree=2)),
                ("linear", QuantileRegressor(
                    solver="highs-ds",
                    alpha=0,
                )),
            ])
        elif method == "lgbm_quant":
            model = LGBMRegressor(
                objective='quantile',
                alpha=0.5,
                random_state=random_state,
                min_data_in_bin=1,
                min_data_in_leaf=1
            )
        else:
            return -1
    else:
        if method == "lin":
            model = Pipeline([
                ("poly", PolynomialFeatures(degree=2)),
                ("linear", LinearRegression())]
            )
        elif method == "lin_lasso":
            model = Pipeline([
                ("poly", PolynomialFeatures(degree=2)),
                ("lasso", Lasso())]
            )
        else:
            model = estimators[method]()

    if conformity_score == "Gamma":
        conf_score = GammaConformityScore()
    else:
        conf_score = AbsoluteConformityScore()

    params = STRATEGIES[strategy]

    param_space = create_param_grid(method, len(train_x.columns))

    # check if 10 features are available, elsewise use 9-fold cross validation
    k = 10 if len(train_x) > 10 else 9

    # generate experiment
    selection = GridSearchCV(
        model,
        param_space,
        n_jobs=-1,
        verbose=1,
        cv=k,
        scoring=make_scorer(mean_absolute_percentage_error, greater_is_better=False),
    )

    # Scale Data

    scaler_train = StandardScaler()
    #scaler_calib = StandardScaler()
    #scaler_test = StandardScaler()

    scaler_train.fit(train_x)
    train_x = scaler_train.transform(train_x)

    if strategy == "cqr":
        # scaler_calib.fit(calib_x)
        calib_x = scaler_train.transform(calib_x)

    # scaler_test.fit(test_x)
    test_x = scaler_train.transform(test_x)

    with mlflow.start_run() as run:
        try:
            logging.info(f"Start hyperparam search using: {str(param_space)}")
            start = time.perf_counter_ns()

            with parallel_backend("threading"):
                selection.fit(train_x, train_y)

            logging.info(f"Finished Grid Search, Start Conformal Prediction")
            if strategy == "cqr":
                estimator = selection.best_estimator_

                print("Shapes: ")
                print(train_x.shape)
                print(calib_x.shape)

                cqr_mapie = MapieQuantileRegressor(estimator, method='quantile', cv='split',
                                                   alpha=alpha)
                cqr_mapie.fit(
                    train_x, train_y,
                    X_calib=calib_x, y_calib=calib_y,
                    random_state=random_state
                )
            else:
                best_params = selection.best_params_
                if method == "lin":
                    model = Pipeline([
                        ("poly", PolynomialFeatures(degree=best_params['poly__degree'])),
                        ("linear", LinearRegression())]
                    )
                elif method == "lin_lasso":
                    model = Pipeline([
                        ("poly", PolynomialFeatures(degree=best_params['poly__degree'])),
                        ("lasso", Lasso(alpha=best_params['lasso__alpha']))]
                    )
                else:
                    model = estimators[method](**best_params)

                mapie_model = MapieRegressor(  # type: ignore
                    model,
                    conformity_score=conf_score,
                    agg_function="median",
                    n_jobs=-1,
                    **params
                )
                with parallel_backend("threading"):
                    mapie_model.fit(train_x, train_y)
            end = time.perf_counter_ns()
            mlflow.log_metric("learning_time", (end - start) * 0.000000001)
            mlflow.sklearn.log_model(selection.best_estimator_, "")
            mlflow.log_params(selection.best_params_)
            mlflow.log_metric("best_score", selection.best_score_)
            logging.info("Predict on test set and save to cache.")
            if strategy == "cqr":
                prediction = _make_pred_artifact(cqr_mapie.predict(test_x), test_y)
            else:
                prediction = _make_pred_artifact(mapie_model.predict(test_x, alpha=[alpha]), test_y)

        except Exception as e:
            logging.error(f"During learning the following error occured: {e}")
            raise e
        finally:
            if logs_to_artifact:
                mlflow.log_artifact("logs.txt", "")

    # prediction[0] = scaler_y.inverse_transform(prediction[0])
    return prediction
