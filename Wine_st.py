import streamlit as st
import pandas as pd
import catboost as CatBoostClassifier #type:ignore
import numpy as np
from joblib import load # Para cargar el modelo entrenado
import joblib  # Para cargar el modelo 
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler #type:ignore

#--------------------------------------------------------------------------------------------------#

# Función para cargar el modelo y el transformador 
def load_components():
    model = joblib.load('best_model.joblib')
    yeo = joblib.load('yeo_transf.joblib')  
    scaler = joblib.load('robust_scaler.joblib') 
   
    return model, yeo, scaler 

# Cargar el modelo y el transformador
model, yeo, scaler  = load_components()

#--------------------------------------------------------------------------------------------------#

#Interfaz
st.title("Calidad del vino")
st.subheader("Introduce las características del vino para predecir su calidad:")

# Cuadros para completar los valores de medición de los vinos
user_input = {
    'volatile_acidity': st.text_input('Volatile acidity', placeholder="Ingresar medición...", 
                                      help="Sugerencia: el rango debe ser entre 0.12 y 1.58"),
    'citric_acid': st.text_input('Citric acid', placeholder="Ingresar medición...",
                                  help="Sugerencia: el rango debe ser entre 0.0 y 1.0"),
    'alcohol': st.text_input('Alcohol', placeholder="Ingresar medición...", 
                             help="Sugerencia: el rango debe ser entre 8.40000 y 14.90"),
    'fixed_acidity': st.text_input('Fixed acidity', placeholder="Ingresar medición...", 
                                   help="Sugerencia: el rango debe ser entre 4.60 y 15.90"),
    'sulphates': st.text_input('Sulphates', placeholder="Ingresar medición...", 
                               help="Sugerencia: el rango debe ser entre 0.33 y 2.0"),
    'total_sulfur_dioxide': st.text_input('Total sulfur dioxide', placeholder="Ingresar medición...", 
                                          help="Sugerencia: el rango debe ser entre 6.0 y 289.0"),
    'free_sulfur_dioxide': st.text_input('Free sulfur dioxide', placeholder="Ingresar medición...", 
                                          help="Sugerencia: el rango debe ser entre 1.0 y 68.0"),                                                                           
}

# Convertir los valores ingresados a DataFrame
user_df = pd.DataFrame([user_input])
st.write("Valores ingresados:", user_df)

#--------------------------------------------------------------------------------------------------#

# Columnas a transformar y eliminar después de la transformación
transf_col = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
              'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'pH',
              'sulphates', 'alcohol', 'density']
colum_elim = ['pH', 'density', 'residual_sugar', 'chlorides']

# Asegurar que las columnas necesarias están presentes y en el orden correcto
for col in transf_col:
    if col not in user_df:
        user_df[col] = 0

user_df = user_df[transf_col]

#--------------------------------------------------------------------------------------------------#

#Botón para realizar la predicción
if st.button('Predecir'):

    # Convertir las entradas del usuario de texto a números
    user_df = user_df.apply(pd.to_numeric, errors='coerce')
   
    # Verificar que no haya valores NaN (valores no ingresados)
    if user_df.isnull().values.any():
        st.error("Por favor, ingresa todos los valores correctamente.")
    else: 
        
    # Aplicar transformación Yeo Johnson 
        transf_yeo = yeo.transform(user_df)

    # Convertir de nuevo a DataFrame     
        df_col_elim = pd.DataFrame(transf_yeo, columns=transf_col)

    # Eliminar las columnas irrelevantes
        df_fin = df_col_elim.drop(columns=colum_elim)

    # Aplicar RobustScaler a los datos finales  
        df_scaler = scaler.transform(df_fin)  

    # Convertir de nuevo a DataFrame    
        df_scaler = pd.DataFrame(df_scaler, columns=df_fin.columns)
   
    # Mostrar los datos finales antes de la predicción
        st.write("Datos finales para predicción:",df_scaler)

        # Realizar predicción de calidad del vino
        try:
            
            prediction = model.predict(df_scaler)

            if prediction[0] == 0:
                st.write('Calidad del vino: Baja')
            else:
                st.write('Calidad del vino: Alta')

        #Errores        
        except AttributeError as e:
            st.error(f"Error al predecir: {e}")

#--------------------------------------------------------------------------------------------------#

# Función para reestablecer los valores
def reset_input():
    for key in st.session_state.keys():
        del st.session_state[key]

# Botón para reestablecer
if st.button('Reestablecer'):
    reset_input()
    st.experimental_rerun()

#--------------------------------------------------------------------------------------------------#

# >>>  terminal: streamlit run Wine_st.py

