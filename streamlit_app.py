import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Ruta al archivo CSV
file_path = "/content/BBDD TA - BD.csv"

def cargar_transformar_datos(df):
    """
    Carga y transforma los datos de un DataFrame.
    """
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

    return df_original, df_low_corr, df_res

def entrenar_modelo(df_low_corr):
    """
    Entrena un modelo Random Forest con los datos proporcionados.
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

    return best_random_forest_model

def predecir_nuevos_datos(modelo, df_nuevos_transformados):
    """
    Realiza predicciones con el modelo entrenado y los nuevos datos.
    """
    predicciones = modelo.predict(df_nuevos_transformados)
    return predicciones

def main():
    # Cargar datos iniciales
    df_original = pd.read_csv(file_path, sep=',', header=0, index_col=0)
    df_original, df_low_corr, df_res = cargar_transformar_datos(df_original)

    # Entrenar modelo con los datos disponibles
    modelo = entrenar_modelo(df_low_corr)

    st.title('Predicción de Clústeres')

    st.write("Introduce los datos para predecir el clúster:")

    # Formularios de entrada de datos
    rol = st.selectbox('Rol', ['Comercial'])
    pais = st.selectbox('PAIS', ['COLOMBIA'])
    empresa = st.selectbox('EMPRESA', ['BROKERS'])
    salario_bruto = st.number_input('Salario Bruto', min_value=0)
    cantidad_transacciones = st.number_input('Cantidad de Transacciones', min_value=0)
    meta = st.number_input('Meta', min_value=0)
    nivel = st.selectbox('NIVEL', ['PRIMARIA', 'BACHILLER', 'TECNICO', 'TECNÓLOGO', 'PREGRADO', 'POSTGRADO'])
    fecha_ingreso = st.date_input('Fecha de Ingreso')
    fecha_retiro = st.date_input('Fecha de Retiro', value=pd.to_datetime('today'))
    salario_referente = st.number_input('Salario Referente', min_value=0)
    grupo_escala = st.text_input('Grupo Escala')
    complejidad = st.selectbox('Complejidad', ['Baja', 'Media', 'Alta'])
    ta = st.number_input('TA', min_value=0)
    escolaridad = st.selectbox('ESCOLARIDAD', ['PRIMARIA', 'BACHILLER', 'TECNICO', 'TECNÓLOGO', 'PREGRADO', 'POSTGRADO'])
    sede = st.text_input('SEDE')
    hijos = st.selectbox('HIJOS', ['No', 'Sí'])
    estado_civil = st.selectbox('ESTADO CIVIL', ['Soltero', 'Casado', 'Divorciado'])
    genero = st.selectbox('GENERO', ['Masculino', 'Femenino'])
    fuente_reclutamiento = st.selectbox('Fuente de Reclutamiento', ['LinkedIn', 'Indeed', 'Otros'])
    tipo_contacto = st.selectbox('Tipo de Contacto', ['Directo', 'Indirecto'])

    if st.button('Predecir'):
        nuevos_datos = pd.DataFrame({
            'Rol': [rol],
            'PAIS': [pais],
            'EMPRESA': [empresa],
            'SALARIO_BRUTO': [salario_bruto],
            'Cantidad de Transacciones': [cantidad_transacciones],
            'Meta': [meta],
            'NIVEL': [nivel],
            'FECHA DE INGRESO': [fecha_ingreso],
            'FECHA DE RETIRO': [fecha_retiro],
            'SALARIO_REFERENTE': [salario_referente],
            'GRUPO ESCALA': [grupo_escala],
            'Complejidad': [complejidad],
            'TA': [ta],
            'ESCOLARIDAD': [escolaridad],
            'SEDE': [sede],
            'HIJOS': [hijos],
            'ESTADO_CIVIL': [estado_civil],
            'GENERO': [genero],
            'Fuente de Reclutamiento': [fuente_reclutamiento],
            'Tipo de Contacto': [tipo_contacto]
        })

        df_nuevos_transformados = cargar_transformar_datos(pd.concat([df_original, nuevos_datos], ignore_index=True))[1].tail(1)

        # Realizar predicciones
        predicciones = predecir_nuevos_datos(modelo, df_nuevos_transformados)
        
        st.write(f'Predicción del clúster: {predicciones[0]}')

if __name__ == '__main__':
    main()
