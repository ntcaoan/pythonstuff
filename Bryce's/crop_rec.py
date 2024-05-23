import pandas as pd
from keras import layers, Sequential

crop_rec_data = pd.read_csv('../Data/Crop_Recommendation.csv')

print(crop_rec_data.head().to_string())
