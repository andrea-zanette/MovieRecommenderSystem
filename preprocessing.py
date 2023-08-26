import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import concurrent.futures
import multiprocessing
import time

NEGATIVE_POINTS_TRAIN = 4
NEGATIVE_POINTS_TEST = 99
BATCH_SIZE = 1024  # With 1024 process took 08:41 min, with 2048 process took 08:18 min
DEST_PATH = "data"


def process_batch(df, ids, num_movies):
    batch_train = []
    batch_test = []
    for user_id in ids:
        # Extract all the samples for each user and sort the rows by the timestamp values
        df_user = df[df["userId"] == user_id].sort_values("timestamp")

        # Get for each user the oldest and the latest rate
        time_min = df_user["timestamp"].min()
        time_max = df_user["timestamp"].max()
        if time_min == time_max:  # In case of a user had seen only 1 movie
            time_max += 1

        # Sample random idxs of movies that the user haven't seen
        negative_data_idx = np.random.choice(np.setdiff1d(np.arange(num_movies), df_user["movieId"].to_numpy()),
                                             NEGATIVE_POINTS_TEST + NEGATIVE_POINTS_TRAIN, replace=False)

        # Generate the negative samples at different timestamps
        negative_data = pd.DataFrame({
            "userId": user_id,
            "movieId": negative_data_idx,
            "rating": 0.,
            "timestamp": np.random.randint(time_min, time_max, size=len(negative_data_idx))
        },
            columns=df.columns
        )

        # N - 1 samples are determined to be in the training set
        batch_train.append(df_user.iloc[:-1])
        # Nth sample is determined to be in the testing set
        batch_test.append(df_user.iloc[-1:])
        # Add negative samples to training and testing sets
        batch_train.append(negative_data.iloc[:NEGATIVE_POINTS_TRAIN])
        batch_test.append(negative_data.iloc[NEGATIVE_POINTS_TRAIN:])

    return batch_train, batch_test


if __name__ == "__main__":
    np.random.seed(0)

    # Import dataset from the .csv file
    df = pd.read_csv("./ml-25m/ratings.csv")

    # Convert the explicit data to implicit data
    df["rating"] = 1.0

    # Encode user and movies
    user_ids = df["userId"].unique().tolist()
    movie_ids = df["movieId"].unique().tolist()

    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}

    df["userId"] = df["userId"].map(user2user_encoded)
    df["movieId"] = df["movieId"].map(movie2movie_encoded)

    num_users = len(user_ids)
    num_movies = len(movie_ids)

    print("Number of users: {}, Number of Movies: {}".format(num_users, num_movies))

    # Init the .csv files containing training and testing data
    if not os.path.exists(DEST_PATH):
        os.makedirs(DEST_PATH)
    df_train = pd.DataFrame(columns=df.columns)
    df_test = pd.DataFrame(columns=df.columns)
    df_train.to_csv(os.path.join(DEST_PATH, "train.csv"), header=True, mode="w", index=False)
    df_test.to_csv(os.path.join(DEST_PATH, "test.csv"), header=True, mode="w", index=False)

    max_processes = multiprocessing.cpu_count()
    # Generate training and testing sets + Add negative data points
    batches_indexes = [np.arange(num_users)[i:i + BATCH_SIZE] for i in range(0, num_users, BATCH_SIZE)]
    pbar = tqdm(total=len(batches_indexes))
    start_t = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
        for result in executor.map(process_batch, [df] * len(batches_indexes), batches_indexes,
                                   [num_movies] * len(batches_indexes)):
            pd.concat(result[0], ignore_index=True).to_csv(os.path.join(DEST_PATH, "train.csv"), index=False, mode="a",
                                                           header=False)
            pd.concat(result[1], ignore_index=True).to_csv(os.path.join(DEST_PATH, "test.csv"), index=False, mode="a",
                                                           header=False)
            pbar.update(1)
    pbar.close()
    finish_t = time.perf_counter()
    execution_time = finish_t - start_t
    # Format elapsed time as MM:SS
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)
    formatted_time = f"{minutes:02}:{seconds:02}"
    print("Execution time:", formatted_time)
