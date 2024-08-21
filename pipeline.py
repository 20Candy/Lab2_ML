# Librerías para manipulación de datos
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport

# Librerías para visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Librerías para preprocesamiento / limpieza de datos
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

# Librerías para modelado
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pickle


# Pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


# ------------------------------------------

# Cargar dataset
data = pd.read_csv('./datasets/test.csv')


# Definir las columnas numéricas y categóricas
numerical_columns = [
    'LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
    'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces',
    'GarageYrBlt', 'GarageCars', 'GarageArea'
]
categorical_columns = [
    'Neighborhood', 'ExterQual', 'Foundation', 'BsmtFinType1'
]
processed_categorical = [
    'Neighborhood_NridgHt', 'ExterQual_Gd',	'Foundation_PConc', 'BsmtFinType1_GLQ'
]
data = data[numerical_columns + categorical_columns]




# Define el preprocesamiento de columnas categóricas
def preprocess_categorical(df, processed_columns):
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded[processed_columns]

# Crear transformadores para las columnas numéricas y categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),         # Imputar valores faltantes
])

categorical_transformer = Pipeline(steps=[
    ('preprocessor', FunctionTransformer(           # Procesar columnas categóricas
        func=lambda df: preprocess_categorical(df, processed_categorical),
        validate=False
    ))
])




# Crear un preprocesador con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ],
)




# Cargar el modelo previamente entrenado desde un archivo
with open('./models/modelo_best.pkl', 'rb') as archivo_entrada:
    model = pickle.load(archivo_entrada)




# Función para hacer predicciones usando el modelo cargado
def predict_with_preprocessing(data):

    data_preprocessed = preprocessor.fit_transform(data)
    predictions = model.predict(data_preprocessed)
    return predictions

# Realizar predicciones
output = predict_with_preprocessing(data)




# Convertir las predicciones a DataFrame y guardarlas en un archivo CSV
pd.DataFrame(output, columns=['Predictions']).to_csv('./datasets/predictions.csv', index=False)
