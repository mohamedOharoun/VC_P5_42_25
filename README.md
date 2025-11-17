# Visión por computador - Práctica V
## Autores
 - Juan Carlos Rodríguez Ramírez
 - Mohamed O. Haroun Zarkik

## Introducción

## Entorno y librerías
Para el funcionamiento de esta práctica será necesario tener un entorno con las siguientes dependencias:

```bash
conda create --name VC_P5 python=3.11.5
conda activate VC_P5

pip install opencv-python matplotlib imutils mtcnn tensorflow deepface tf-keras cmake

pip install scikit-learn
pip install scikit-image
```

## Tarea I

Se pide desarrollar un prototipo biométrico, es decir, entrenar un modelo extrayendo características biométricas de las imágenes y detectar algo en función de las extraídas. Algo así como un discriminador. En este caso, se ha realizado un discriminador binario para detectar si una persona tiene o no barba. Para ello, se emplearon los datasets de [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) (200.000+ imágenes de celebridades) y [Beard or No Beard](https://www.kaggle.com/datasets/uzair01/beard-or-no-beard) (~300 imágenes discriminando famosos con barba más notable y sin). Del primer dataset, se hizo una selección manual de unas 270 imágenes de personas con barba y sin barba, buscando que no fuesen barbas muy variantes y sobre todo notables. El punto de la tarea es que detecte barba, no que se confunda con la cantidad de tamaños de barba, colores y formas (bigotes, candado, perilla...). 

En un principio, el modelo parecía necesitar más imágenes; se comenzó entrenando el modelo con un 95% de PCA, HOG y LBP, con un tamaño de imágenes reescalado a 128x128 píxeles. Tras probar el modelo tanto con la cámara como con inferencias, no se consiguió detectar ninguna barba, lo cual podía indicar falta de imágenes. 

No obstante, se optó por subir la calidad del reescalado a 192x192 píxeles, con la intención de que con más calidad, el resultado fuese mejor. No se obtuvo mejora alguna, lo que empezó a mosquearnos, ya que haciendo inferencias con las propias imágenes de entrenamiento, no se conseguía la detección de personas con barbas. Después de mucho razonamiento y una pequeña charla con el profesor Modesto Castrillón, se consiguió detectar el error, y se solventaron otros problemas relacionados.   

### Problema detectado

Durante las pruebas se observó que el clasificador:

- Apenas detectaba casos **con barba**, incluso sobre imágenes del propio conjunto de entrenamiento.  
- Funcionaba aceptablemente en validación cruzada (k‑fold), pero se comportaba casi como un clasificador constante en la demo (webcam e inferencias). 

#### Resultados de validación cruzada (5 folds)

| Esquema        | Accuracy Fold 1 | Accuracy Fold 2 | Accuracy Fold 3 | Accuracy Fold 4 | Accuracy Fold 5 | Accuracy media |
|----------------|-----------------|-----------------|-----------------|-----------------|-----------------|----------------|
| PCA + SVM      | 0.71            | 0.72            | 0.71            | 0.73            | 0.73            | 0.72           |
| HOG + SVM      | 0.70            | 0.60            | 0.68            | 0.68            | 0.72            | 0.68           |
| LBP + SVM      | 0.76            | 0.72            | 0.71            | 0.81            | 0.79            | 0.76           |

***

Analizando el código se identificaron dos causas principales:

1. **Incoherencia entre entrenamiento e inferencia (falta de escalado):**  
   - En el entrenamiento, los descriptores (PCA, HOG, LBP) se normalizaban con `MinMaxScaler` antes de entrenar el SVM.  
   - En las inferencias (imagen estática y webcam) se calculaba HOG/LBP, pero se pasaban directamente al SVM sin aplicar el mismo escalado.  
   - El modelo recibía vectores en un rango distinto al de entrenamiento, lo que explica que, incluso con imágenes conocidas, tendiera a una sola clase.

2. **Recorte de cara con poco contexto en la demo:**  
   - El dataset de entrenamiento contiene caras con cierto contexto (parte de cabeza, cuello y algo de fondo).  
   - En la demo, Viola–Jones devolvía un bounding box muy ajustado a la cara, que después se reescalaba directamente, perdiendo parte de la silueta y del contexto que el modelo había visto durante el entrenamiento.  
   - Esta diferencia de distribución entre las imágenes de entrenamiento y las de la webcam degradaba aún más el rendimiento.

***

### Cambios realizados

#### 1. Coherencia entrenamiento–inferencia: guardar y aplicar el scaler

**Antes:**

- En el entrenamiento:  
  ```python
  scaler = MinMaxScaler()
  train_X = scaler.fit_transform(X_train)
  test_X  = scaler.transform(X_test)
  clf = GridSearchCV(SVC(...))
  clf = clf.fit(train_X, y_train)
  ```
- En la inferencia:  
  - Se calculaba el descriptor LBP/HOG y se llamaba a `svm_lbp.predict(desc)` sin normalizar.

**Posteriormente:**

1. **Entrenamiento: la función de predicción devuelve también el scaler**

   ```python
   def GetPredictions(X_train, X_test, y_train, y_test):
       scaler = MinMaxScaler()
       train_X = scaler.fit_transform(X_train)
       test_X  = scaler.transform(X_test)

       parameters = {'C': [1e3, 5e3],
                     'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01]}
       clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                          parameters, cv=5)
       clf = clf.fit(train_X, y_train)

       y_pred = clf.predict(test_X)
       return y_pred, y_test, clf, scaler
   ```

2. **En el bucle k‑fold se guarda también el scaler del mejor modelo (en nuestro caso, LBP+SVM):**

   ```python
   y_pred_lbp_svm, y_test_lbp_svm, svm_lbp, scaler_lbp = GetPredictions(
       Xlbp3x3_train, Xlbp3x3_test, y_train, y_test
   )

   if fold == 1:
       joblib.dump(svm_lbp,   "svm_lbp_barba.joblib")
       joblib.dump(scaler_lbp,"scaler_lbp_barba.joblib")
   ```

3. **Inferencias: se aplica el mismo scaler antes de `predict`**

   ```python
   svm_lbp    = joblib.load("svm_lbp_barba.joblib")
   scaler_lbp = joblib.load("scaler_lbp_barba.joblib")

   def preprocess_lbp_from_gray(gray):
       gray_resized = cv2.resize(gray, (WIDTH, HEIGHT),
                                 interpolation=cv2.INTER_AREA)
       feat_lbp = lbphist(gray_resized,
                          ncellsx=3, ncellsy=3,
                          width=WIDTH, height=HEIGHT,
                          lbp_method="nri_uniform")
       desc = feat_lbp.astype("float32").reshape(1, -1)
       desc_scaled = scaler_lbp.transform(desc)
       return desc_scaled
   ```

   - Esta función se utiliza tanto en las inferencias sobre imágenes del dataset como en el flujo de *webcam*.  
   - Con esto, el SVM recibe exactamente el mismo tipo de datos (mismo rango y distribución) que durante el entrenamiento.

**Efecto:** Tras este cambio, el modelo vuelve a comportarse en inferencia de forma coherente con los resultados de k‑fold y es capaz de clasificar correctamente muchas imágenes con barba, incluidas las del propio dataset.

***

#### 2. Aumento del bounding box para incluir contexto

**Antes:**

- En la demo con webcam:  
  - Viola–Jones detectaba una cara con `(x, y, w, h)`.  
  - Se recortaba exactamente ese rectángulo y se reescalaba a `WIDTH × HEIGHT` para calcular LBP.  
  - El modelo había sido entrenado con imágenes donde la cara aparecía con algo más de contexto (cabeza/cuello/fondo), por lo que la distribución de píxeles no coincidía.

**Posteriormente:**

1. **Se define una función para expandir el bounding box alrededor de la cara:**

   ```python
   def expand_bbox(x, y, w, h, frame_width, frame_height, factor=1.3):
       cx = x + w / 2.0
       cy = y + h / 2.0
       new_w = w * factor
       new_h = h * factor

       x_new = max(0, int(cx - new_w / 2.0))
       y_new = max(0, int(cy - new_h / 2.0))
       x_new2 = min(frame_width,  int(x_new + new_w))
       y_new2 = min(frame_height, int(y_new + new_h))

       return x_new, y_new, x_new2 - x_new, y_new2 - y_new
   ```

2. **En el bucle de la webcam se usa el bbox expandido y el mismo preprocesado que en entrenamiento:**

   ```python
   faces = face_cascade.detectMultiScale(frame_gray,
                                         scaleFactor=1.3,
                                         minNeighbors=5)

   for (x, y, w, h) in faces:
       x_exp, y_exp, w_exp, h_exp = expand_bbox(
           x, y, w, h, w_frame, h_frame, factor=1.3
       )

       face_gray = frame_gray[y_exp:y_exp+h_exp, x_exp:x_exp+w_exp]

       desc_scaled = preprocess_lbp_from_gray(face_gray)
       pred  = int(svm_lbp.predict(desc_scaled)[0])
       label = classlabels[pred]
       ...
   ```

El dataset de entrenamiento contiene caras con contexto, no recortes extremadamente ajustados, por lo que LBP está captando textura de barba pero también forma global de la mandíbula, cuello y parte del fondo. Al ampliar el bounding box, las imágenes de la webcam se parecen más a las del entrenamiento (tipo de recorte, proporción de cara/fondo), reduciendo el desajuste entre ambos dominios.

**Efecto:** Con el bounding box expandido y el preprocesado coherente, el sistema pasa de prácticamente no detectar barbas en tiempo real a producir predicciones razonables, en línea con las métricas de validación cruzada.

*** 

### Resultados de la tarea I

En la validación cruzada k‑fold se entrenan `k` modelos distintos por esquema, pero para el despliegue solo es necesario conservar un modelo representativo.

La validación cruzada se utiliza para estimar de forma fiable el rendimiento medio de cada combinación de características y clasificador (PCA, HOG, LBP + SVM), no para almacenar simultáneamente todos los modelos generados en los distintos folds. En cada fold se entrena un SVM con su correspondiente `MinMaxScaler` y, dado que las particiones son estratificadas y el conjunto de datos está equilibrado, los resultados obtenidos son muy similares entre folds (accuracy en el intervalo 0.70–0.80). 

Por simplicidad, se decidió guardar el modelo y el scaler obtenidos en el primer fold, que se consideran representativos del comportamiento global observado en los experimentos, mientras que el resto de folds se emplean únicamente para calcular métricas y analizar la variabilidad. En un escenario de producción, una vez identificado el mejor esquema (en este caso LBP + SVM), lo más adecuado sería reentrenar un único modelo con todo el conjunto de datos y desplegar ese modelo final, pero para esta práctica resulta suficiente utilizar el modelo del primer fold y mantener el código más sencillo. El cuarto fold es el mejor posible, pero la diferencia no es tan significativa.  

En conjunto, los resultados muestran que el modelo LBP + SVM es capaz de diferenciar personas con y sin barba en la mayoría de los casos, obteniendo un rendimiento superior al de las alternativas basadas en PCA y HOG tanto en accuracy como en f1‑score.

#### f1-score medio por clase (aprox.)

| Esquema    | con_barba (f1) | sin_barba (f1) |
|------------|----------------|----------------|
| PCA + SVM  | ≈ 0.72         | ≈ 0.72         |
| HOG + SVM  | ≈ 0.69         | ≈ 0.66         |
| LBP + SVM  | ≈ 0.77         | ≈ 0.74         |

#### GIF demostración

<img src="./prototipo_biometrico.gif" alt="demo" width="600" />

