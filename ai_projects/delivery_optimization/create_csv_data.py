import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

#Generate dummy data
n = 1000
np.random.seed(42)
start_date = datetime(2025,1,1,9) #9 AM time to start delivery
date_range_days = (datetime(2025,6,30) - start_date).days

data = {
    'order_id': range(1, n+1),
    'customer_lat': np.random.uniform(23.5, 24.5, n),
    'customer_lon': np.random.uniform(58.5,59.5,n),
    'delivery_window_start': [
        start_date + timedelta(days=random.randint(0,date_range_days), minutes=random.randint(0,180))
        for _ in range(n)
        ],
    'delivery_duration_estimate': np.random.randint(10,40,n),
    'vehicle_capacity': np.random.choice([1,2],n),
    'package_size': np.random.choice([1,2],n),
}
df = pd.DataFrame(data)
df['delivery_window_end'] = df['delivery_window_start'] + pd.to_timedelta(df['delivery_duration_estimate'], unit = 'm')

#Save in CSV
df.to_csv('SAVE_CSV_PATH', index=False) #Add path to save generated CSV