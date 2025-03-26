import pandas as pd
import numpy as np

# Crear una DataFrame con valores de la matriz de confusión
confusion_matrix = pd.DataFrame({
    "TP": [40],  # True Positives
    "TN": [30],  # True Negatives
    "FP": [20],  # False Positives
    "FN": [10]   # False Negatives
})

# Calcular métricas utilizando pandas y numpy
confusion_matrix["Accuracy"] = (confusion_matrix["TP"] + confusion_matrix["TN"]) / \
                               (confusion_matrix["TP"] + confusion_matrix["TN"] + confusion_matrix["FP"] + confusion_matrix["FN"])

confusion_matrix["Precision"] = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FP"])

confusion_matrix["Recall"] = confusion_matrix["TP"] / (confusion_matrix["TP"] + confusion_matrix["FN"])

confusion_matrix["F1-Measure"] = 2 * (confusion_matrix["Precision"] * confusion_matrix["Recall"]) / \
                                  (confusion_matrix["Precision"] + confusion_matrix["Recall"])

# Mostrar los resultados
print(confusion_matrix[["Accuracy", "Precision", "Recall", "F1-Measure"]])
