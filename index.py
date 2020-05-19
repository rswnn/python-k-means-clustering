import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pltfrom
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

retail = pd.read_csv("/Users/riswanardiansah/Desktop/gempa.csv")
retail.head()

retail.info()

ritel_x = retail.iloc[:, 5:7]
ritel_x.head()


sns.scatterplot(x="KM", y="Mag", data=retail, s=50, color="red", alpha=0.5)
pltfrom.show()

x_array = np.array(ritel_x)
print(x_array)

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)

kmeans = KMeans(n_clusters=5, random_state=123)
kmeans.fit(x_scaled)
print(kmeans.cluster_centers_)

print(kmeans.labels_)
retail["kluster"] = kmeans.labels_
retail.head()

fig, ax = pltfrom.subplots()
sct = ax.scatter(x_scaled[:, 1], x_scaled[:, 0], s=100,
                 c=retail.kluster, marker="o", alpha=0.5)
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 1], centers[:, 0], c='blue', s=200, alpha=0.5)
pltfrom.title("Hasil Klustering K-Means")
pltfrom.xlabel("Scaled KM")
pltfrom.ylabel("Scaled Mag")
pltfrom.show()
