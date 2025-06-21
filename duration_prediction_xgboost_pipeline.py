# %%
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import pickle

# %%
import mlflow

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('nyc_greentaxi_2021_tripdata_experiment') 

# %%
def read_dataframe(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime)
    df['duration'] = df['duration'].apply(lambda td: td.total_seconds()/60)
    df = df[((df['duration'] > 1) & (df['duration'] <= 60))]
    
    categorical = ['PULocationID','DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df

# %%
df_train = read_dataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-01.parquet')
df_val = read_dataframe('https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet')

# %%
categorical = ['PU_DO']
numerical = ['trip_distance']

dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
val_dicts = df_val[categorical + numerical].to_dict(orient='records')

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

y_train = df_train['duration'].values
y_val = df_val['duration'].values

# %%
from math import sqrt

# %% [markdown]

# %%
import xgboost as xgb

# %%
import mlflow.xgboost
from mlflow.models.signature import infer_signature
mlflow.xgboost.autolog(disable=True)

# %%
with mlflow.start_run():
    train = xgb.DMatrix(X_train, label=y_train)
    val = xgb.DMatrix(X_val, label=y_val)

    mlflow.set_tag("engineer", "richkinwe")
    mlflow.log_param("train_data_path", "green_tripdata/green_tripdata_2021-01.parquet")
    mlflow.log_param("val_data_path", "green_tripdata/green_tripdata_2021-02.parquet")
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

    with open('models.preprocessor.b', 'wb') as f_out:
        pickle.dump(dv, f_out)

    mlflow.log_artifact('models.preprocessor.b', artifact_path='preprocessor')

    mlflow.xgboost.log_model(booster, artifact_path='models_xgboost_mlflow')