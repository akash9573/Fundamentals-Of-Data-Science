import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
data = {
    'age': [25, 30, 35, 40, 45, 50, 55, 60],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'blood_pressure': [120, 122, 118, 130, 128, 135, 140, 138],
    'cholesterol': [180, 200, 190, 220, 210, 240, 230, 250],
    'outcome': ['Good', 'Bad', 'Good', 'Bad', 'Good', 'Bad', 'Good', 'Bad']
}
df = pd.DataFrame(data)
df_encoded = pd.get_dummies(df, columns=['gender'], drop_first=True)

X = df_encoded.drop("outcome", axis=1)
y = df_encoded["outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


k = 3  
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="Good")
recall = recall_score(y_test, y_pred, pos_label="Good")
f1 = f1_score(y_test, y_pred, pos_label="Good")
conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
