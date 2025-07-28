import pandas as pd
from pandas import read_csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data_frame = read_csv("minute_weather.csv")
# print(data_frame.shape)
temp_df = data_frame.values[:100]
# print (temp_df)
season_df = pd.DataFrame(temp_df, columns=data_frame.columns)
# print(season_df)

# Dropping Nan values
print(season_df.isnull().sum())
season_df.dropna(inplace=True)
print(season_df.isnull().sum())

# Dropping rowID column
season_df.drop(axis=1, labels=['rowID', 'max_wind_direction', 'max_wind_speed',
                               'min_wind_direction', 'min_wind_speed', 'hpwren_timestamp'], inplace=True)
print(season_df.head())

inertia = []

for i in range(1, 100):
    kmeans = KMeans(i, init="k-means++")
    kmeans.fit(season_df)
    inertia.append(kmeans.inertia_)


print(inertia)
plt.figure(figsize=(16, 5))
plt.title("Elbow Graph of K from 1 to 100")
plt.plot(range(1, 100), inertia, marker='o', linestyle='--', markersize=3)
plt.grid(True)
plt.show()
