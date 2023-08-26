import pandas as pd
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers.legacy import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from model.recommender_model import RecommenderModel
from data_generator import DataGenerator

DATA_PATH = "data/train.csv"
LOG_PATH = "logs"
MODEL_PATH = "model/weights"
BATCH_SIZE = 256

# Import dataset and initialise DataGenerator
train_df = pd.read_csv(DATA_PATH, usecols=["userId", "movieId", "rating"])
num_users = len(train_df["userId"].unique())
num_movies = len(train_df["movieId"].unique())
positive_data = train_df[train_df["rating"] == 1].groupby("userId").head(300)  # Limit number of + samples per user
negative_data = train_df[train_df["rating"] == 0]
gen = DataGenerator(positive_data, negative_data, batch_size=BATCH_SIZE, pos_frac=1 / 2)

# Free up memory
del train_df
del positive_data
del negative_data

# Get the model
model = RecommenderModel.get_model(num_users, num_movies, batch_size=BATCH_SIZE)
model.compile(optimizer=Adam(learning_rate=0.01),
              loss=MeanSquaredError(),
              metrics=[MeanAbsoluteError()])

# Callbacks
checkpoint = ModelCheckpoint(os.path.join(MODEL_PATH, '{epoch:02d}.hdf5'), verbose=1)
tensorboard = TensorBoard(log_dir=LOG_PATH, histogram_freq=1)
callbacks = [checkpoint, tensorboard]

# Train the model
model.fit(gen, epochs=5, steps_per_epoch=gen.__len__(), callbacks=callbacks)
