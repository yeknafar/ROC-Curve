import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve
from matplotlib import pyplot
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)

ns_probs = np.zeros(shape = len(testy))

inputs = layers.Input(shape = (20,))
x = layers.Dense(20, activation = "relu")(inputs)
x = layers.Dense(10, activation = "relu")(x)
out = layers.Dense(1, activation = "sigmoid")(x)

net = models.Model(inputs, out)

net.compile(optimizer=  "sgd",
              loss = "binary_crossentropy",
              metrics = ["accuracy"])

h = net.fit(trainX, trainy, batch_size=32, epochs=15)

preds = net.predict(testX).ravel()
float_preds = np.rint(preds)
round_pred = float_preds.astype("int")
print(round_pred)
tn, fp, fn, tp = confusion_matrix(testy, round_pred))

auc_mlp = roc_auc_score(testy, preds)
auc_random = roc_auc_score(testy, ns_probs)
print(np.zeros(shape = testy.shape))
print("auc MLP: ", auc_mlp)
print("auc random: ", auc_random)

random_fpr, random_tpr, _ = roc_curve(testy,ns_probs)
mlp_fpr, mlp_tpr, _ = roc_curve(testy, preds)

plt.plot(random_fpr, random_tpr, linestyle="-", label = "random")
plt.plot(mlp_fpr, mlp_tpr, linestyle="--", label = "mlp")
plt.legend()
plt.show()

precision, recall, th = precision_recall_curve(testy, preds)
