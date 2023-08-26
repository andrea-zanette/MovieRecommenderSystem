from keras.utils import Sequence
import math
import pandas as pd

RAND_SEED = 0


class DataGenerator(Sequence):
    def __init__(self, positive_data, negative_data, batch_size, shuffle=True, pos_frac=0.5):
        self.positive_data = positive_data
        self.negative_data = negative_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pos_frac = pos_frac
        if self.shuffle:
            self.positive_data = self.positive_data.sample(frac=1, random_state=RAND_SEED).reset_index(drop=True)

    def __len__(self):
        return math.ceil(len(self.positive_data) / (self.batch_size * self.pos_frac))

    def __getitem__(self, idx):
        pos_low = int(idx * (self.batch_size * self.pos_frac))
        pos_high = int(min(pos_low + (self.batch_size * self.pos_frac), len(self.positive_data)))
        positive_batch = self.positive_data[pos_low:pos_high]
        negative_batch = self.negative_data.sample(int(self.batch_size * (1 - self.pos_frac)))
        batch = pd.concat([positive_batch, negative_batch], ignore_index=True).sample(frac=1, random_state=RAND_SEED)
        return [batch["userId"].to_numpy(), batch["movieId"].to_numpy()], batch["rating"].to_numpy()

    def on_epoch_end(self):
        if self.shuffle:
            self.positive_data = self.positive_data.sample(frac=1, random_state=RAND_SEED).reset_index(drop=True)
