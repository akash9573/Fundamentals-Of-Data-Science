import pandas as pd
import numpy as np
from scipy import stats
np.random.seed(42) 
ratings = np.random.randint(1, 6, 100)
df = pd.DataFrame({'rating': ratings})
average_rating = df['rating'].mean()
std_dev = df['rating'].std()
confidence_level = 0.95
sample_size = len(df)
margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * (std_dev / np.sqrt(sample_size))
confidence_interval = (average_rating - margin_of_error, average_rating + margin_of_error)
print(f"Average Rating: {average_rating:.2f}")
print(f"Confidence Interval ({confidence_level * 100}%): {confidence_interval}")
