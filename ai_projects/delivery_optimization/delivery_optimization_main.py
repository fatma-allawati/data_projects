import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv('YOUR_CSV_PATH') #Add CSV path

#Clustering delivery locations using KMeans
coords = df[['customer_lat','customer_lon']]

#Apply kmean clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['route_cluster'] = kmeans.fit_predict(coords)

#plot cluster
fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(df['customer_lon'], df['customer_lat'], c=df['route_cluster'], cmap='viridis', s=50)
legend = ax.legend(*scatter.legend_elements(), title="Cluster")
ax.set_title('Delivery Location Clusters')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.tight_layout()

#Save figure
plot_path = "SAVE_FIGURE_PATH" #Save your figure path
fig.savefig(plot_path)
plot_path


###################################################################################################################################
###################################################################################################################################
#############################Train Estimated Time of Arrival (ETA) Prediction mModel###############################################
###################################################################################################################################
###################################################################################################################################

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np
from math import radians, cos, sin, asin, sqrt

df['delivery_window_start'] = pd.to_datetime(df['delivery_window_start'])
df['hour'] = df['delivery_window_start'].dt.hour

#Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371 #Earth Radians in KM
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

#Compute distance between warehouse and customer
warehouse_lat, warehouse_lon = 23.6, 58.5  # replace with your real warehouse location

#Calculate distance from warehouse to customer for each row
df['distance_km'] = df.apply(lambda row: haversine(warehouse_lat, warehouse_lon, row['customer_lat'], row['customer_lon']), axis=1)

#Define traffic levels based on delivery hour
def estimate_traffic(hour):
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        return 'High'
    elif 10 <= hour <= 15:
        return 'Medium'
    else:
        return 'Low'

df['traffic_level'] = df['hour'].apply(estimate_traffic)

#Optional: Convert map traffic levels to numeric if needed for modeling
traffic_map = {'Low': 1, 'Medium': 2, 'High': 3}
df['traffic_level'] = df['traffic_level'].map(traffic_map)

x = df[['hour','package_size','vehicle_capacity','route_cluster','customer_lat','customer_lon','distance_km','traffic_level']]
y = df['delivery_duration_estimate']


#Train/test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#Train model
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(x_train,y_train)

#Calculate root mean squared error (RMSE) to evaluate model performance
y_pred = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(f"Model RMSE: {rmse:.2f}")


###################################################################################################################################
###################################################################################################################################
#########################################Scatter plot of actual vs predicted#######################################################
###################################################################################################################################
###################################################################################################################################

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='teal')
plt.plot([min(y_test), max(y_test)],[min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel("Actual Delivery Duration (minutes)")
plt.ylabel("Predicted Delicery Duration (minutes)")
plt.title("Actual vs Predicted Delivery Duration")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()