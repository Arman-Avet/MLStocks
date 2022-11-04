from hyperopt import tpe, hp, Trials
from hyperopt.fmin import fmin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

seed = 42

PATH = 'C:/Users/A changer/PycharmProjects/ML/data/dataset.csv'

data = pd.read_csv(PATH)

# Dropping null values in column 'text' and 'sm'
data = data[['text', 'sm', 'price + 1']]
# data = data.loc[data['sm'] == '$GOOGL']
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
y = data['price + 1']
X = data.drop(['price + 1'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=seed)
string_features = 'text'
categorical_features = ['sm']


def objective(params):
    loss = str(params['loss'])
    alpha = float(params['alpha'])
    penalty = str(params['penalty'])
    max_iter = int(params['max_iter'])

    preprocessor = ColumnTransformer(
        transformers=[('str', TfidfVectorizer(stop_words='english', max_features=3500), string_features),
                      ('cat', OneHotEncoder(handle_unknown='ignore'), ['sm'])])

    model = SGDRegressor(alpha=alpha, loss=loss, penalty=penalty, max_iter=max_iter)

    regressor = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)
                                ])
    regressor.fit(X_train, y_train)

    preds = regressor.predict(X_test)

    score = mean_absolute_error(y_test, preds)

    return score


def optimize(trial):
    params={'loss':hp.choice('loss', ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
           'alpha':hp.uniform('alpha', 1e-4, 0.001),
           'penalty':hp.choice('penalty',['l2', 'l1', 'elasticnet']),
           'max_iter':hp.quniform('max_iter',1000, 2000, 1)}
    best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=3,rstate=np.random.default_rng(seed))
    return best


trial = Trials()
best = optimize(trial)
preprocessor = ColumnTransformer(
    transformers=[('str', TfidfVectorizer(stop_words='english', max_features=3500), string_features),
                  ('cat', OneHotEncoder(handle_unknown='ignore'), ['sm'])])

model = SGDRegressor(alpha=best['alpha'],
                     loss=['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'][best['loss']],
                     penalty=['l2', 'l1', 'elasticnet'][best['penalty']],
                     max_iter=best['max_iter'])

regressor = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', model)
                            ])
regressor.fit(X_train, y_train)

preds = regressor.predict(X_test)

score = mean_absolute_error(y_test, preds)

print("MAE: ", score)

