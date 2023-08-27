# MovieRecommenderSystem
This project implements a Movie Recommender System using neural collaborative filtering techniques from [this](https://link.springer.com/article/10.1007/s41870-022-00858-4) paper.

It provides personalized movie recommendations based on user interactions and preferences. 

## Table of Contents

- [Getting Started](#getting-started)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Getting Started

1. Clone this repository
2. Download the [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) from Kaggle and save it in a folder named `ml-25m` in the root directory.
  
## Usage

1. Preprocess the dataset: `python preprocess.py`
2. Train the model: `python train.py`
3. Test the model: `python test.py`
4. (Optional) See model architecture: `python recommender_model.py`

## Dependencies

- Python 3.9
- TensorFlow 2.13.0 (TensorFlow-Metal 1.0.1 for ARM Mac)
- Pandas
- NumPy

## License

This project is licensed under the [MIT License](LICENSE).
