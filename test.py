from keras.models import load_model
import pandas as pd
from tqdm import tqdm

MODEL_PATH = "model/weights/05.h5"
DATA_PATH = "data/test.csv"
HIT_RANK = 10

# Load model and dataset
model = load_model(MODEL_PATH)
test_df = pd.read_csv(DATA_PATH, usecols=["userId", "movieId", "rating"])

# Predict
print("Compute predictions")
pred = model.predict([test_df["userId"].to_numpy(), test_df["movieId"].to_numpy()])
test_df["prediction"] = pred
print(pred)

# Compute hit ratio
print("Compute hit ratio")
user_ids = test_df['userId'].unique()
hits = 0
for user_id in tqdm(user_ids):
    user_df = test_df[test_df["userId"] == user_id].sort_values("prediction")
    user_df = user_df.reset_index(drop=True)
    row_id = user_df.index[user_df["rating"] == 1.0]
    if row_id < HIT_RANK:
        hits += 1
print("Hit ratio", (hits / len(user_ids)) * 100, "%")
