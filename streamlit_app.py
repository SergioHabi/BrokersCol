import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
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

# Ruta al archivo CSV
file_path = "BBDD TA - BD.csv"

def cargar_transformar_datos(file_path):
    """
    Carga y transforma los datos del archivo CSV.

    Parameters:
    file_path (str): Ruta al archivo CSV.

    Returns:
    tuple: DataFrame original, DataFrame con baja correlación y DataFrame resampleado.
    """
    df = pd.read_csv(file_path, sep=',', header=0, index_col=0)
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
    scaler = StandardScaler()
    df['CVR_estandarizada'] = scaler.fit_transform(df[['CVR']])

    kbd = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    df['CVR_binned'] = kbd.fit_transform(df[['CVR_estandarizada']])

    # Clustering
    X = df['CVR'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
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
    imputador_knn = KNNImputer(n_neighbors=5)
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

    return df_original, df_low_corr, df_res, scaler, kbd, kmeans

@st.cache_data
def entrenar_modelo(df_low_corr):
    """
    Entrena un modelo Random Forest con los datos proporcionados.

    Parameters:
    df_low_corr (DataFrame): DataFrame con baja correlación.

    Returns:
    tuple: Modelo entrenado, precisión del modelo, reporte de clasificación, datos de prueba y predicciones.
    """
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

def cargar_modelo_y_transformadores():
    """
    Carga el modelo Random Forest y los transformadores desde los archivos .pkl.

    Returns:
    tuple: Modelo entrenado, escalador, KBinsDiscretizer y KMeans.
    """
    with open('modelo_random_forest.pkl', 'rb') as file:
        modelo = pickle.load(file)
    with open('escalador.pkl', 'rb') as file:
        escalador = pickle.load(file)
    with open('kbd.pkl', 'rb') as file:
        kbd = pickle.load(file)
    with open('kmeans.pkl', 'rb') as file:
        kmeans = pickle.load(file)
    return modelo, escalador, kbd, kmeans

def transformar_datos_usuario(df_usuario, escalador, kbd, kmeans):
    """
    Transforma los datos ingresados por el usuario.

    Parameters:
    df_usuario (DataFrame): Datos del usuario.
    escalador (StandardScaler): Escalador para la variable CVR.
    kbd (KBinsDiscretizer): Discretizador para la variable CVR estandarizada.
    kmeans (KMeans): Modelo de clustering KMeans.

    Returns:
    DataFrame: Datos del usuario transformados.
    """
    df_usuario['CVR'] = df_usuario['Cantidad de Transacciones'] / df_usuario['Meta']
    df_usuario['CVR'] = df_usuario['CVR'].fillna(0)
    df_usuario['CVR_estandarizada'] = escalador.transform(df_usuario[['CVR']])
    df_usuario['CVR_binned'] = kbd.transform(df_usuario[['CVR_estandarizada']])
    df_usuario['CVR_cluster'] = kmeans.predict(df_usuario[['CVR']])
    return df_usuario

def main():
    st.title('Predicción de Calidad de Nuevos Ingresos')
    st.write('Esta aplicación predice la calidad de nuevos ingresos para la compañía.')

    # Cargar datos y entrenar el modelo
    df_original, df_low_corr, _, scaler, kbd, kmeans = cargar_transformar_datos(file_path)
    modelo, accuracy, report, X_test, y_test, y_pred = entrenar_modelo(df_low_corr)

    st.subheader('Datos Transformados')
    st.write(df_low_corr)

    st.subheader('Mejores Hiperparámetros')
    st.write(modelo.get_params())

    st.subheader('Exactitud del Modelo')
    st.write(accuracy)

    st.subheader('Informe de Clasificación')
    st.text(report)

    st.subheader('Matriz de Confusión')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.sidebar.header('Parámetros del Candidato')
    input_data = {}
    for col in df_original.columns:
        if df_original[col].dtype == 'object':
            options = df_original[col].dropna().unique()
            input_data[col] = st.sidebar.selectbox(f'Selecciona {col}', options=options)
        else:
            input_data[col] = st.sidebar.number_input(f'{col}', value=0.0)

    input_df = pd.DataFrame(input_data, index=[0])

    st.subheader('Datos del Candidato Ingresados')
    st.write(input_df)

    if st.button('Predecir Calidad'):
        input_df_transformed = transformar_datos_usuario(input_df, scaler, kbd, kmeans)
        resultado = modelo.predict(input_df_transformed[['CVR_cluster']])
        st.subheader('Resultado de la Predicción')
        st.write('La calidad del nuevo ingreso es:', resultado[0])

if __name__ == '__main__':
    main()
