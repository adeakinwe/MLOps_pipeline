import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import pickle

import mlflow

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('nyc_greentaxi_2021_tripdata_experiment')
mlflow.xgboost.autolog(disable=True)

import xgboost as xgb
import mlflow.xgboost

from math import sqrt
from pathlib import Path

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


def read_dataframe(year, month):
    #url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    url = f'green_tripdata/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime)
    df['duration'] = df['duration'].apply(lambda td: td.total_seconds() / 60)
    df = df[((df['duration'] > 1) & (df['duration'] <= 60))]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']

    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv, year, month):
    with mlflow.start_run() as run:
        run_id = run.info.run_id  # Capture run_id

        train = xgb.DMatrix(X_train, label=y_train)
        val = xgb.DMatrix(X_val, label=y_val)

        year_val = year if month < 12 else year + 1
        month_val = month + 1 if month < 12 else 1

        mlflow.set_tag("engineer", "richkinwe")
        mlflow.log_param("train_data_path", f"nyc_taxi_green_tripdata{year}-{month:02d}.parquet")
        mlflow.log_param("val_data_path", f"nyc_taxi_green_tripdata{year_val}-{month_val:02d}.parquet")
        mlflow.set_tag("model", "xgboost")

        best_params = {
            'max_depth': 30,
            'learning_rate': 0.0959,
            'reg_alpha': 0.0181,
            'reg_lambda': 0.0117,
            'min_child_weight': 1.0606,
            'objective': 'reg:linear',
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=50,
            evals=[(val, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(val)
        rmse = sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)

        with open('models/models.preprocessor.b', 'wb') as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact('models/models.preprocessor.b', artifact_path='preprocessor')

        mlflow.xgboost.log_model(booster, artifact_path='models_xgboost_mlflow')

        print(f'Run completed. RMSE: {rmse}, Run ID: {run_id}')
        return run_id  # Return run_id


def run(year, month):
    print(f'Starting training for {year}-{month:02d}')

    df_train = read_dataframe(year=year, month=month)

    year_val = year if month < 12 else year + 1
    month_val = month + 1 if month < 12 else 1

    df_val = read_dataframe(year=year_val, month=month_val)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    run_id = train_model(X_train, y_train, X_val, y_val, dv, year, month)
    print(f'Training completed. Run ID: {run_id}')
    return run_id


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a taxi ride duration prediction model.")
    parser.add_argument('--year', type=int, required=True, help="Year of the data to train on.")
    parser.add_argument('--month', type=int, required=True, help="Month of the data to train on.")
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)
    print(f'Use this Run ID to register or promote the model: {run_id}')