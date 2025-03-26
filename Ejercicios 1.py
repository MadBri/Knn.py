# Knn.py
```python
   from sklearn.neighbors import KNeighborsClassifier
   import numpy as np

   # Datos de entrenamiento
   data = {
    "Punto X": [2, 4, 1, 2, 2, 2, 3, 3],
    "Punto Y": [0, 4, 1, 4, 2, 3, 4, 3],
    "Clase": [0, 1, 0, 1, 0, 1, 0, 1]
}

   # Definir clasificador
   knn = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
   knn.fit(X_train, y_train)

   # Clasificar el caso
   prediction = knn.predict([[2.5, 2.5]])
   print(f"Clase predicha: {prediction[0]}")
   ```
