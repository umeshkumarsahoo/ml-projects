import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE = 'housing_model.pkl'
PIPELINE_FILE = 'housing_pipeline.pkl'

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline=Pipeline([
        ('imputer',SimpleImputer(strategy="mean")),
        ('scaler',StandardScaler()),
    ])
    cat_pipeline=Pipeline([
        ('one_hot',OneHotEncoder(handle_unknown='ignore'))
    ])
    full_pipeline=ColumnTransformer(
        [
            ("num",num_pipeline,num_attribs),
            ("cat",cat_pipeline,cat_attribs),
        ]
    )
    return full_pipeline
if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
    housing = pd.read_csv('housing.csv')
    housing['income_cat'] = pd.cut(housing['median_income'],
                                   bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                   labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        start_train_set = housing.iloc[train_index].drop("income_cat", axis=1)
        start_test_set = housing.iloc[test_index].drop("income_cat", axis=1)
    housing = start_train_set.copy()
    df= housing.copy().to_csv('input.csv', index=False)
    housing_labels = housing['median_house_value'].copy()
    housing.drop('median_house_value', axis=1, inplace=True)
    num_attribs = list(housing.drop("ocean_proximity", axis=1).columns)
    cat_attribs = ["ocean_proximity"]
    pipeline=build_pipeline(num_attribs,cat_attribs)
    housing_prepared = pipeline.fit_transform(housing)
    print(housing_prepared.shape)
    model=RandomForestRegressor()
    model.fit(housing_prepared,housing_labels)
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("model is trainer and saved")
else:
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)
    input_data=pd.read_csv('housing.csv')
    input_data=pd.read_csv('input.csv')
    transform_input=pipeline.transform(input_data)
    predictions=model.predict(transform_input)
    input_data['median_house_value']=predictions
    input_data.to_csv('predictions.csv', index=False)
    print('the result is done and saved to predictions.csv')
