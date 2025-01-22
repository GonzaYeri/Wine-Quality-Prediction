# Wine-Quality-Prediction (Classification project)

## Descripción
Este proyecto busca predecir la calidad del vino (alta '1' o baja '0') basándose en características químicas de éste. El modelo de clasificación utiliza técnicas avanzadas de preprocesamiento de datos y algoritmos de aprendizaje automático, integrados en una aplicación web interactiva desarrollada con Streamlit.

## Características Principales
- **Aplicación interactiva:** Una interfaz sencilla e intuitiva para ingresar las características del vino y obtener predicciones.
- **Modelo de Machine Learning:** CatBoostClassifier, entrenado con datos procesados adecuadamente.
- **Preprocesamiento de datos:** Transformaciones Yeo-Johnson y escalado robusto para mejorar la calidad del análisis.

## Contenido del Repositorio
- **`Wine_st.py`:** Código fuente de la aplicación Streamlit.
- **`Wine.ipynb`:** Notebook con el análisis exploratorio de datos (EDA) y el entrenamiento del modelo.
- **Archivos del modelo:**
  - `best_model.joblib`
  - `yeo_transf.joblib`
  - `robust_scaler.joblib`

## Instalación
1. Clona este repositorio:
   ```bash
   git clone https://github.com/GonzaYeri/Wine-Quality-Prediction.git
   cd Wine-Quality-Prediction
   ```
2. Instala las dependencias requeridas:
   ```bash
   pip install -r requirements.txt
   ```
3. Asegúrate de que los archivos del modelo (`best_model.joblib`, `yeo_transf.joblib`, `robust_scaler.joblib`) estén en el mismo directorio que `Wine_st.py`.

## Uso
1. Ejecuta la aplicación:
   ```bash
   streamlit run Wine_st.py
   ```
2. Introduce las características del vino en los campos proporcionados.
3. Haz clic en **Predecir** para determinar si la calidad del vino es alta o baja.
4. Usa el botón **Reestablecer** para limpiar los campos y realizar una nueva predicción.

## Estructura del Código
- **`Wine_st.py`:**
  - Carga de modelos y transformadores preentrenados.
  - Interfaz de usuario creada con Streamlit.
  - Funciones para preprocesamiento y predicción.
- **`Wine.ipynb`:**
  - Análisis exploratorio de datos (EDA).
  - Entrenamiento y evaluación del modelo CatBoostClassifier.

## Licencia
Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más información.

## Autor
Gonzalo Sebastián Yeri
