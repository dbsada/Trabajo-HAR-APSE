# Reconocimiento de Actividades con Acelerómetro

Este proyecto es un trabajo de la asignatura **Aplicaciones Sectoriales (APSE)** que aborda la clasificación automática de actividades físicas a partir de señales inerciales registradas con un acelerómetro triaxial. El objetivo es comparar distintos enfoques de modelado, desde métodos clásicos hasta arquitecturas profundas, evaluando su capacidad para generalizar a sujetos no vistos durante el entrenamiento.

## Estructura del proyecto

> [!Warning]
> Durante la realización de la práctica se interpretó erróneamente que el eje Z representaba la altura. Sin embargo, era el eje Y el que tenía este significado. Esto afecta al modelo [cnn.ipynb](cnn.ipynb) y, para solucionarlo, se debe modificar la función `get_height_signal_z_only` ubicada en el notebook:
> ```
> h_z = get_height_signal_z_only(z, fs)
> # cambiar por:
> h_y = get_height_signal_y_only(y, fs)
> ```
> Este cambio no se ha implementado, ya que requeriría modificaciones en múltiples partes del código y el informe, y el proyecto se ha considerado finalizado en su estado actual.

- **data/**: Contiene los archivos `train.csv` y `test.csv` con los datos originales.
- **artifacts/**: Pesos, checkpoints y predicciones generadas por los modelos.
- **images/**: Figuras y esquemas utilizados en los notebooks e informe.

- **[informe.pdf](informe.pdf)**: Documento principal del informe en LaTeX.
- **utils.py**: Funciones auxiliares y utilidades comunes.

- **[EDA.ipynb](EDA.ipynb)**: Análisis exploratorio de datos (EDA) detallado.
- **[rf.ipynb](rf.ipynb)**: Notebook para el modelo Random Forest.
- **[cnn1d.ipynb](cnn1d.ipynb)**: Notebook para el modelo CNN 1D y CNN 1D + LSTM.
- **[transformer.ipynb](transformer.ipynb)**: Notebook para el modelo Transformer.
- **[cnn.ipynb](cnn.ipynb)**: Notebook para el modelo CNN dual-stream.
- **[global_metrics.ipynb](global_metrics.ipynb)**: Notebook para el análisis de métricas globales y comparación de modelos.

## Uso de los notebooks

Cada notebook está diseñado para ser independiente y autocontenido, permitiendo al lector ejecutar y comprender cada modelo sin necesidad de consultar los demás. Todos los notebooks, excepto el de Random Forest, incluyen una versión reducida del análisis exploratorio de datos (EDA) al inicio, para facilitar la comprensión del pipeline completo desde cero.

**Importante:**
- Para el notebook de Random Forest (`rf.ipynb`), se recomienda encarecidamente revisar primero el notebook de EDA (`EDA.ipynb`), ya que el análisis y la ingeniería de características manuales se explican en detalle allí y no se repiten en el notebook del modelo.
- El resto de notebooks (CNN, CNN1D, Transformer) contienen la parte básica del EDA (carga y visualización de datos) para que puedan ser leídos y ejecutados de forma independiente.

## Requisitos

- Python 3.8+
- PyTorch
- scikit-learn
- pandas, numpy, matplotlib, seaborn
- Jupyter Notebook

Instala las dependencias necesarias con:

```bash
pip install -r requirements.txt
```

## Autores
Antonio Álvarez, Diego Besada, Natalia Corchón y Alfonso Jimena 
