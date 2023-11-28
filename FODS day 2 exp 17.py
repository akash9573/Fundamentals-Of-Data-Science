import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
np.random.seed(42)
customer_ids = np.arange(1, 101)
amount_spent = np.random.normal(loc=50, scale=10, size=100)
visit_frequency = np.random.poisson(lam=5, size=100)


transaction_data = pd.DataFrame({'customer_id': customer_ids, 'amount_spent': amount_spent, 'visit_frequency': visit_frequency})


features = transaction_data[['amount_spent', 'visit_frequency']]


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()
optimal_k = int(input("enter the K value:"))

kmeans_model = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
cluster_labels = kmeans_model.fit_predict(scaled_features)
transaction_data['cluster'] = cluster_labels
cluster_means = transaction_data.groupby('cluster').mean()

print(cluster_means)
