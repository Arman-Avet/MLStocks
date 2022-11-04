from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
import pandas as pd

# random seed
seed = 42

PATH = 'C:/Users/A changer/PycharmProjects/ML/data/dataset.csv'

data = pd.read_csv(PATH)
print(data.head())

# on enlève les valeurs nul de 'text' et 'sm'
data = data[['text', 'sm', 'price + 1']]
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

# déf. de X et y
y = data['price + 1']
X = data.drop(['price + 1'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=seed)

# ########## MODÈLE ###############
model = LinearRegression(n_jobs=-1)
# #################################
string_features = 'text'
categorical_features = ['sm']


preprocessor = ColumnTransformer(transformers=
                                 [('str', TfidfVectorizer(stop_words='english', max_features=3500),
                                   string_features), ('cat',  OneHotEncoder(handle_unknown='ignore'), ['sm'])])
# Pipeline
regressor = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Entrance X et y
regressor.fit(X_train, y_train)

# print précision
print('Training MAE:', mean_absolute_error(regressor.predict(X_train), y_train))

# Predictions des valeurs test
preds = regressor.predict(X_test)

# Évaluation du modèle
score = mean_absolute_error(y_test, preds)
print('Validation MAE:', score)
