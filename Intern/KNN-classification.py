import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

digits = datasets.load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred))

n_samples = 10
fig, axes = plt.subplots(1, n_samples, figsize=(10, 3), subplot_kw={'xticks':[], 'yticks':[]})
for ax, image, label, prediction in zip(axes, X_test[:n_samples], y_test[:n_samples], y_pred[:n_samples]):
    ax.imshow(image.reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.text(0, 7, f'True: {label}\nPred: {prediction}', color='red' if label != prediction else 'green')
plt.suptitle('Sample Digits with Predictions')
plt.show()
