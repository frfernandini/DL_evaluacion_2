Clasificación de Imágenes de Prendas de Vestir con Redes Neuronales Convolucionales (CNN)
Introducción
Este proyecto aborda el problema de clasificación de imágenes utilizando el dataset Fashion-MNIST. El objetivo principal es implementar y evaluar diferentes arquitecturas de Redes Neuronales Convolucionales (CNN) para clasificar 70,000 imágenes en escala de grises (28x28 píxeles) en 10 categorías distintas de prendas de vestir.

Carga y Preprocesamiento de Datos
El dataset Fashion-MNIST se carga utilizando tensorflow.keras.datasets. Se divide en 60,000 imágenes para entrenamiento y 10,000 para prueba. Las categorías son:

0: Camiseta/Top
1: Pantalón
2: Suéter
3: Vestido
4: Abrigo
5: Sandalia
6: Camisa
7: Zapatilla
8: Bolso
9: Botín
Pasos de Preprocesamiento:
Reshape: Las imágenes se transforman a un formato de 4 dimensiones (num_muestras, altura, ancho, canales) para ser compatibles con las capas convolucionales (en este caso, (60000, 28, 28, 1) y (10000, 28, 28, 1)).
Normalización: Los valores de los píxeles se escalan de [0, 255] a [0, 1] dividiendo por 255.0. Esto ayuda a estabilizar y acelerar el entrenamiento del modelo.
One-Hot Encoding: Las etiquetas categóricas se convierten a un formato de vectores binarios de longitud 10 para la clasificación multiclase.
Modelos Implementados y Resultados
Se exploraron y evaluaron seis modelos diferentes, cada uno introduciendo mejoras o cambios en la arquitectura o el proceso de entrenamiento. La función post_entrenamiento se utilizó para estandarizar la evaluación, guardado, y visualización de resultados (pérdida, precisión, reporte de clasificación y matriz de confusión).

Modelo 1: CNN Base con Optimizador SGD
Arquitectura: CNN secuencial con dos bloques Conv2D (32 y 64 filtros de 3x3) + MaxPooling2D, Flatten, y capas Dense (64, 10). activation='relu' en capas ocultas, activation='softmax' en la salida.
Optimizador: SGD con learning_rate=0.01.
Épocas: 15.
Observación: Convergencia lenta e inestable debido al SGD no adaptativo. Dificultad para clasificar prendas similares (e.g., Camisa).
Modelo 2: CNN Base con Optimizador Adam
Arquitectura: Idéntica al Modelo 1.
Optimizador: Adam con learning_rate=0.001.
Épocas: 15.
Observación: Aprendizaje mucho más rápido, pero con mayor inestabilidad y sobreajuste (overfitting) significativo a partir de la época 4, evidenciado por una brecha creciente entre las curvas de entrenamiento y validación.
Modelo 3: CNN Optimizador Adam con Dropout
Arquitectura: Idéntica al Modelo 2, con la adición de una capa Dropout(0.5) después de la primera capa Dense.
Optimizador: Adam con learning_rate=0.001.
Épocas: 15.
Observación: El Dropout redujo drásticamente el sobreajuste y estabilizó las curvas de aprendizaje, cerrando la brecha entre entrenamiento y validación. El modelo aprende patrones más generales.
Modelo 4: Profundidad + Padding
Arquitectura: Modelo 3 extendido con padding='same' en todas las capas Conv2D y la adición de una tercera capa Conv2D (128 filtros de 3x3) y una capa Dense (128 neuronas).
Optimizador: Adam con learning_rate=0.001.
Épocas: 15.
Observación: Logró el mayor Accuracy bruto. El padding ayudó a preservar información de los bordes. Sin embargo, la mayor complejidad reintrodujo un ligero sobreajuste a partir de la época 9.
Modelo 5: Data Augmentation + Early Stopping
Arquitectura: Modelo 4 con capas de Data Augmentation (RandomFlip, RandomRotation, RandomZoom) añadidas al inicio.
Optimizador: Adam con learning_rate=0.001.
Épocas: 50 (con Early Stopping).
Callbacks: EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True).
Observación: Modelo definitivo. El Data Augmentation mejoró la capacidad de generalización y el Early Stopping detuvo el entrenamiento en el punto óptimo, mitigando el sobreajuste y logrando un val_loss muy bajo.
Modelo 6: Striding (Reemplazando Max-Pooling)
Arquitectura: Modelo 5, pero reemplazando las capas MaxPooling2D por capas Conv2D con strides=(2, 2).
Optimizador: Adam con learning_rate=0.001.
Épocas: 50 (con Early Stopping).
Callbacks: EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True).
Observación: Introdujo mayor inestabilidad en las curvas de validación. Aunque mantuvo un rendimiento similar, se concluyó que para imágenes de baja resolución (28x28), Max-Pooling preserva mejor la información que el submuestreo aprendido con Strides.
Tabla Comparativa de Experimentos
--- ESTUDIO DE ABLACIÓN: EVOLUCIÓN ARQUITECTURA CNN ---
                      Modelo      Épocas       LR  Batch Size Val Accuracy Val Loss  Observación Principal
0        Exp 1: CNN Base (SGD)          15     0.01          64     86.25%   0.3773  Convergencia segura sin sobreajuste, pero muy lenta. Dificultad severa para clasificar prendas similares (clase Camisa).
1                Exp 2: Adam          15    0.001          64     91.04%   0.2881  Aprendizaje rápido. Rompe la barrera del 90% gracias a la inercia adaptativa, pero memoriza los datos generando un claro Overfitting.
2          Exp 3: Dropout (0.5)          15    0.001          64     90.85%   0.2538  El Dropout elimina la memorización. Las curvas de pérdida se juntan y estabilizan, curando el sobreajuste del modelo anterior.
3  Exp 4: Profundidad + Padding          15    0.001          64     92.45%   0.2634  Tercera capa de 128 filtros capta mejor los detalles. Logra el mejor Accuracy bruto, pero el exceso de capacidad induce sobreajuste en la época 9.
4  Exp 5: Data Augmentation + Early Stopping  50 (Freno Auto)    0.001          64     91.93%   0.2404  MODELO DEFINITIVO. Inmunidad total al ruido visual; sacrifica mínima precisión por máxima generalización. El Early Stopping captura el punto óptimo.
5         Exp 6: Strides (Sin MaxPool)  50 (Freno Auto)    0.001          64     91.24%   0.2488  Rendimiento decreciente. Demuestra empíricamente que, a baja resolución (28x28), el MaxPooling preserva mejor los bordes espaciales que el submuestreo aprendido.
Conclusión
El Modelo 5 (Data Augmentation + Early Stopping) se identificó como el mejor modelo del proyecto. Aunque el Modelo 4 logró una precisión bruta ligeramente superior, el Modelo 5 ofreció la mejor capacidad de generalización con un val_loss más bajo (0.2404) y una robustez significativamente mayor frente al sobreajuste.

La combinación de Dropout(0.5), Padding='same', Data Augmentation y Early Stopping permitió que el modelo aprendiera características significativas de las prendas, mejorando notablemente el Recall para la categoría
