
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Ruta al archivo CSV
file_path = "BBDD TA - BD.csv"

# Variables globales para transformaciones
scaler = StandardScaler()
kbd = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
imputador_knn = KNNImputer(n_neighbors=5)

def cargar_transformar_datos(df):
    df = df.replace({',': '.', '-': '/'}, regex=True)
    df = df[(df['Rol'] == 'Comercial') & (df['PAIS'] == 'COLOMBIA') & (df['EMPRESA'] == 'BROKERS')]
    df_original = df.copy()

    # Convertir tipos de datos
    df['SALARIO_BRUTO'] = df['SALARIO_BRUTO'].astype('float64')
    df['Cantidad de Transacciones'] = df['Cantidad de Transacciones'].astype('float64')
    df['Meta'] = df['Meta'].astype('float64')
    df['NIVEL'] = df['NIVEL'].astype('object')
    df['FECHA DE INGRESO'] = pd.to_datetime(df['FECHA DE INGRESO'], format='%d/%m/%Y')
    df['FECHA DE RETIRO'] = pd.to_datetime(df['FECHA DE RETIRO'], format='%d/%m/%Y').fillna(pd.Timestamp('today'))

    # Calcular nuevas características
    df['CVR'] = df['Cantidad de Transacciones'] / df['Meta']
    df['CVR'] = df['CVR'].fillna(0)
    df['Salario_USD'] = np.where(df['PAIS'] == 'COLOMBIA', df['SALARIO_REFERENTE'] / 4000,
                                 np.where(df['PAIS'] == 'MEXICO', df['SALARIO_BRUTO'] / 17, np.nan))
    df['diferencia_dias'] = (df['FECHA DE RETIRO'] - df['FECHA DE INGRESO']).dt.days

    # Estandarización y discretización
    df['CVR_estandarizada'] = scaler.fit_transform(df[['CVR']])
    df['CVR_binned'] = kbd.fit_transform(df[['CVR_estandarizada']])

    # Clustering
    X = df['CVR'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
    df['CVR_cluster'] = kmeans.labels_

    # Eliminar columnas no necesarias
    var_drop = ['GRUPO ESCALA', 'FECHA DE INGRESO', 'FECHA DE RETIRO', 'Rol', 'SALARIO_REFERENTE', 'SALARIO_BASE',
                'SALARIO_BRUTO', 'CVR', 'CVR_estandarizada', 'CVR_binned', 'Complejidad', 'PAIS', 'EMPRESA', 'TA']
    df.drop(var_drop, axis=1, inplace=True)
    df['diferencia_dias'] = df['diferencia_dias'].astype('float64')

    # Codificar escolaridad
    orden_escolaridad = {'PRIMARIA': 0, 'BACHILLER': 1, 'TECNICO': 2, 'TECNÓLOGO': 3, 'PREGRADO': 4, 'POSTGRADO': 5}
    df['ESCOLARIDAD_Numerica'] = df['ESCOLARIDAD'].map(orden_escolaridad)
    df.drop('ESCOLARIDAD', axis=1, inplace=True)

    # Imputación y codificación de datos categóricos
    df_float = df.select_dtypes(include=['float64', 'int32'])
    data_imp = pd.DataFrame(imputador_knn.fit_transform(df_float), columns=df_float.columns, index=df_float.index)

    df_object = df.select_dtypes(include=object)
    datos_dummies = pd.get_dummies(df_object, columns=['SEDE', 'HIJOS', 'ESTADO_CIVIL', 'GENERO', 'NIVEL',
                                                       'Fuente de Reclutamiento', 'Tipo de Contacto'])
    df_com = pd.concat([data_imp, datos_dummies], axis=1)

    # Balanceo de clases
    X = df_com.drop(columns=['CVR_cluster'])
    y = df_com['CVR_cluster']
    smote = SMOTE(sampling_strategy='not majority', random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    df_res = pd.DataFrame(X_res, columns=X.columns)
    df_res['CVR_cluster'] = y_res

    # Filtrar variables con baja correlación
    correlation_matrix = df_res.corr()
    correlation_threshold = 0.5
    low_correlation_vars = correlation_matrix[abs(correlation_matrix['CVR_cluster']) < correlation_threshold]['CVR_cluster']
    low_correlation_var_names = low_correlation_vars.index.tolist()

    if 'CVR_cluster' not in low_correlation_var_names:
        low_correlation_var_names.append('CVR_cluster')

    df_low_corr = df_res[low_correlation_var_names].copy()
    df_low_corr.drop(columns=['Meta', 'diferencia_dias'], axis=1, inplace=True)

    return df_original, df_low_corr, df_res

def agregar_nuevos_datos(df_original, nuevos_datos):
    df_combinado = pd.concat([df_original, nuevos_datos], ignore_index=True)
    return cargar_transformar_datos(df_combinado)

def entrenar_modelo(df_low_corr):
    X = df_low_corr.drop(columns=['CVR_cluster'])
    y = df_low_corr['CVR_cluster']

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Configuración de hiperparámetros para GridSearch
    param_grid = {
        'n_estimators': [50],
        'max_depth': [None],
        'min_samples_split': [5],
        'min_samples_leaf': [2]
    }

    random_forest_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_random_forest_model = RandomForestClassifier(**best_params, random_state=42)
    best_random_forest_model.fit(X_train, y_train)

    y_pred_best = best_random_forest_model.predict(X_test)

    accuracy_best = accuracy_score(y_test, y_pred_best)
    report_best = classification_report(y_test, y_pred_best)

    return best_random_forest_model, accuracy_best, report_best, X_test, y_test, y_pred_best
def predecir_nuevos_datos(modelo, datos_nuevos_transformados, columnas_entrenadas):
    for col in columnas_entrenadas:
        if col not in datos_nuevos_transformados.columns:
            datos_nuevos_transformados[col] = 0
    datos_nuevos_transformados = datos_nuevos_transformados[columnas_entrenadas]
    return modelo.predict(datos_nuevos_transformados)

# Cargar y transformar datos iniciales
df_original = pd.read_csv(file_path, sep=',', header=0, index_col=0)
df_original, df_low_corr, df_res = cargar_transformar_datos(df_original)

# Entrenar el modelo con los datos iniciales
best_model, accuracy, report, X_test, y_test, y_pred = entrenar_modelo(df_low_corr)
st.title('Prediccion de Calidad de Nuevos Ingresos')
st.write('Esta aplicacion predice la calidad de nuevos ingresos para la compañia.')


st.subheader('Datos Transformados')
st.write(df_low_corr)

modelo, accuracy_best, report_best, X_test, y_test, y_pred_best = entrenar_modelo(df_low_corr)

st.subheader('Mejores Hiperparámetros')
st.write(modelo.get_params())

st.subheader('Exactitud del Modelo')
st.write(accuracy_best)

st.subheader('Informe de Clasificación')
st.text(report_best)

st.subheader('Matriz de Confusión')
cm = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Interfaz de Streamlit
st.title("Predicción de CVR Cluster")

# Entradas del usuario para los nuevos datos
st.sidebar.header("Ingresar nuevos datos")


salario_bruto = st.sidebar.number_input("Salario Bruto", min_value=0.0)
salario_referente = st.sidebar.number_input("Salario Referente", min_value=0.0)
escolaridad = st.sidebar.selectbox("ESCOLARIDAD", ["PRIMARIA", "BACHILLER", "TECNICO", "TECNÓLOGO", "PREGRADO", "POSTGRADO"])
hijos = st.sidebar.selectbox("HIJOS", ["No", "Sí"])
estado_civil = st.sidebar.selectbox("ESTADO CIVIL", ["Soltero", "Casado", "Divorciado"])
genero = st.sidebar.selectbox("GENERO", ["Masculino", "Femenino"])

# Botón para hacer la predicción
if st.sidebar.button("Predecir"):
    # Crear DataFrame para los nuevos datos
    nuevos_datos = pd.DataFrame({
        'SALARIO_BRUTO': [salario_bruto],
        'SALARIO_REFERENTE': [salario_referente],
        'ESCOLARIDAD': [escolaridad],
        'HIJOS': [hijos],
        'ESTADO_CIVIL': [estado_civil],
        'GENERO': [genero],
    })

    # Agregar nuevos datos y transformar
    df_original, df_low_corr_nuevos, df_res_nuevos = agregar_nuevos_datos(df_original, nuevos_datos)

    # Preparar datos nuevos para la predicción
    df_nuevos_transformados = df_low_corr_nuevos.drop(columns=['CVR_cluster'])
    columnas_entrenadas = X_test.columns.tolist()  # Columnas usadas en el entrenamiento

    # Realizar predicciones
    predicciones = predecir_nuevos_datos(best_model, df_nuevos_transformados, columnas_entrenadas)
    
    st.subheader('Resultados de la Predicción para Nuevos Datos')
    st.write('Precisión del mejor modelo:')
    st.write(accuracy)

    st.write('Reporte de clasificación:')
    st.text(report)

    st.write('Predicciones para los nuevos datos:')
    st.write(predicciones)
