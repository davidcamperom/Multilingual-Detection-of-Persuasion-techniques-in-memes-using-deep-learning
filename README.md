# Detección Multilingüe de Técnicas de Persuasión en Memes usando técnicas de Deep Learning
**Alumno:** David Campero Maña

**Centro:** Escuela Técnica Superior de Ingeniería. Universidad de Huelva  
**Titulación:** Grado en Ingeniería Informática  
**Departamento:** Ingeniería de la Información y el Conocimiento  
**Grupo de Investigación:** I2C-UHU

**Tutor 1:** Jacinto Mata Vázquez  
**Tutor 2:** Victoria de la Pena Pachón Álvarez

## Descripción del Proyecto

Los memes son uno de los tipos de contenido más utilizados en las campañas de desinformación en línea. Son sobre todo eficaces en las plataformas de redes sociales, ya que allí pueden llegar fácilmente a un gran número de usuarios. En una campaña de desinformación, los memes logran su objetivo de influir en los usuarios mediante una serie de técnicas retóricas y psicológicas, como la simplificación causal, el insulto y la difamación.

El objetivo principal de este trabajo es el estudio e implementación de técnicas para la detección automática de técnicas de persuasión multilingüe en memes, dado por los textos de estos y una instancia que nos dice todas las técnicas de persuasión que se transmiten. Este dataset es una gran opción para detectar persuasión pues proporciona una variedad de argumentos que abordan diferentes temas y perspectivas, ya sean políticas o sociales. Para detectar técnicas de persuasión, hemos utilizado modelos preentrenados como BERT y RoBERTa, entre otros, que son arquitecturas de transformers avanzadas. Estos modelos han sido entrenados en grandes cantidades de datos textuales y son excelentes para una amplia gama de tareas de procesamiento del lenguaje natural. Hemos aprovechado su conocimiento del lenguaje para clasificar técnicas de persuasión.

El trabajo comienza con una contextualización teórica de los conceptos principales de Machine Learning y Deep Learning, así como las estrategias utilizadas para abordar el desequilibrio en la distribución de clases en el conjunto de datos. Estas estrategias incluyen el submuestreo, el procesado de los argumentos para estandarizar el texto y el aumento de datos con back-translation. Estas estrategias han permitido mejorar la capacidad de los modelos para detectar las técnicas de persuasión.

Para evaluar los resultados se participó en “The 18th International Workshop on Semantic Evaluation”, más concretamente en la tarea “SemEval 2024: Multilingual Detection of Persuasion Techniques in Memes”.

En resumen, nuestro enfoque ha sido utilizar modelos preentrenados de vanguardia y técnicas de procesamiento de datos para identificar técnicas de persuasión en memes. Este enfoque ha demostrado ser eficaz en la tarea a resolver.

**Palabras clave:** Desinformación, Memes, Técnicas de Persuasión, Machine Learning, Deep Learning, Procesamiento del Lenguaje Natural, Transformers, BERT, RoBERTa, Backtranslation, SemEval 2024.

## Estructura del Proyecto

La estructura del proyecto es la siguiente:
```
SemEval
├── data
│   ├── dev_gold_labels
│   ├── preprocessed_datasets
│   ├── semeval2024_dev_release
│   ├── scorer-baseline
│   ├── test_data
│   ├── test_data_arabic
│   ├── test_labels_ar_bg_md_version2
├── figures
├── fine_tuned_models
│   ├── distilbert
│   │   ├── distilbert-base-uncased
│   ├── ensemble_by_union
│   ├── ensemble_by_vote
│   ├── FacebookAI
│   │   ├── roberta-base
│   ├── google-bert
│   │   ├── bert-base-uncased
│   ├── microsoft
│   │   ├── deberta-v3-base
│   ├── openai-community
│   │   ├── gpt2
│   ├── xlnet
│   │   ├── xlnet-base-cased
├── notebooks
├── results
├── scripts
```

## Cómo ejecutar la aplicación

Esta guía proporciona un proceso paso a paso para ejecutar el programa. Sigue cada paso cuidadosamente para asegurar una ejecución correcta.

> [!IMPORTANT]  
>Los Jupyter notebooks se encuentran en la carpeta `notebooks`  
>Los scripts de Python se encuentran en la carpeta `scripts`

## 1. Preprocesamiento de Datos

### 1.1. Visualización y Preprocesamiento de Datos
- Abre el Jupyter notebook `data_visualisation_preprocessing.ipynb`.
- Ejecuta todas las celdas del notebook. Este paso incluye tareas de visualización y preprocesamiento de datos.
- Usamos la columna de texto preprocesada para la ampliación de datos.

### 1.2. Reducción y Aumento de Datos
- Ejecuta el script de Python `data_downsampling_augmentation.py`.
```terminal
python3 data_downsampling_augmentation.py -t ./train_df.csv -n 600
```
- Este script realiza la reducción y aumento de datos basándose en los datos preprocesados.

## 2. Ajuste Fino del Modelo y Evaluación
- Procede con el proceso de Fine-Tuning del modelo. Asegúrate de que todas las configuraciones necesarias estén establecidas antes de comenzar el ajuste fino.
- Ejecuta el script `model_fine_tuning_evaluation.py` con el siguiente comando:
```terminal
python3 ./scripts/model_fine_tuning_evaluation.py -train_name original_train -train ./data/preprocessed_datasets/train_df.csv -val ./data/preprocessed_datasets/validation_df.csv -dev ./data/preprocessed_datasets/dev_df.csv -test ./data/preprocessed_datasets/test_df.csv
```

## 3. Ensamblaje
- Abre el Jupyter notebook `perform_ensemble.ipynb`.
- Ejecuta todas las celdas del notebook para realizar el ensamblaje de diferentes modelos.

## 4. Traducir Archivos de Prueba al Inglés
- Abre el Jupyter notebook `translate_test_files_to_en.ipynb`.
- Ejecuta todas las celdas del notebook para traducir los archivos de prueba al inglés.

## 5. Inferencia del Modelo Ajustado Finamente y Evaluación de Pruebas
- Abre el Jupyter notebook `fine_tune_model_inference.ipynb`.
- Ejecuta todas las celdas del notebook para evaluar el modelo en el conjunto de pruebas.

Sigue estos pasos en el orden dado para asegurar una ejecución fluida del programa. Si surgen problemas, consulta los scripts y notebooks respectivos para obtener instrucciones más detalladas.
