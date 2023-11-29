from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
X = np.array([[25, 120], 
              [30, 140], 
              [22, 110]   
             ])
y = np.array([0, 1, 0])  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
new_patient_features = []
for i in range(len(X[0])):
    feature_value = float(input(f"Enter the value for feature {i + 1}: "))
    new_patient_features.append(feature_value)
k_value = int(input("Enter the value of k (number of neighbors): "))
knn_classifier = KNeighborsClassifier(n_neighbors=k_value)
knn_classifier.fit(X_train, y_train)
new_patient_features = np.array(new_patient_features).reshape(1, -1)  # Reshape for a single sample
prediction = knn_classifier.predict(new_patient_features)
if prediction[0] == 0:
    print("The model predicts that the patient does not have the medical condition.")
else:
    print("The model predicts that the patient has the medical condition.")

y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")
