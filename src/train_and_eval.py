######## Steps:
## load the train and test
## train alg
## save the metrices, params

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from get_data import read_params
from urllib.parse import urlparse
import argparse
import mlflow

def eval_metrics(actual, pred):
    f1 = round(f1_score(actual, pred), 3)
    precision = round(precision_score(actual, pred),3)
    recall = round(recall_score(actual, pred),3)
    return f1, precision, recall

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    n_estimators = config["estimators"]["RandomForestClassifier"]["params"]["n_estimators"]
    criterion = config["estimators"]["RandomForestClassifier"]["params"]["criterion"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)


    remote_server_uri = config["mlflow_config"]["remote_server_uri"]
    experiment_name = config["mlflow_config"]["experiment_name"]
    register_model_name = config["mlflow_config"]["registered_model_name"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            criterion=criterion,
            random_state=random_state)
        model.fit(train_x, train_y.values.ravel())

        predicted_qualities = model.predict(test_x)
        
        (f1, precision, recall) = eval_metrics(test_y, predicted_qualities)

        mlflow.log_param("n_estimators",n_estimators)
        mlflow.log_param("criterion", criterion)

        mlflow.log_metric("F1_score",f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model, 
                "modelC", 
                registered_model_name= register_model_name)
        else:
            mlflow.sklearn.load_model(model, "modelC")


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)