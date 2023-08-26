from keras.layers import Input, Embedding, Concatenate, Flatten, Dense, Dropout, Reshape
from keras.models import Model
from keras.regularizers import l2

EMBEDDING_SIZE = 8  # size of the embedding vectors


class RecommenderModel:
    @staticmethod
    def get_model(num_users, num_movies, batch_size):
        # Input layers
        user_input = Input(shape=(1,), name="user_input", batch_size=batch_size)
        movie_input = Input(shape=(1,), name="movie_input", batch_size=batch_size)

        # Embedding layers
        user_embedding = Embedding(input_dim=num_users, output_dim=EMBEDDING_SIZE, name="user_embedding",
                                   embeddings_initializer='he_normal', embeddings_regularizer=l2(1e-6))(user_input)
        movie_embedding = Embedding(input_dim=num_movies, output_dim=EMBEDDING_SIZE, name="movie_embedding",
                                    embeddings_initializer='he_normal', embeddings_regularizer=l2(1e-6))(movie_input)

        # Reshaping
        user_embedding = Reshape((EMBEDDING_SIZE,), name="user_reshaping")(user_embedding)
        movie_embedding = Reshape((EMBEDDING_SIZE,), name="movie_reshaping")(movie_embedding)

        # Concatenate the two embedding layers
        concatenated = Concatenate()([user_embedding, movie_embedding])
        concatenated = Dropout(0.05)(concatenated)

        # Add dense layers
        x = concatenated
        for i in range(6):
            x = Dense(1024 / (2 ** i), kernel_initializer='he_normal', activation='relu',
                      name="dense_layer_" + str(i + 1))(x)
            x = Dropout(0.5)(x)

        # Output layer
        output = Dense(1, kernel_initializer='he_normal', activation='sigmoid', name="output")(x)

        # Create the model
        return Model(inputs=[user_input, movie_input], outputs=output, name="RecommenderModel")


if __name__ == "__main__":
    model = RecommenderModel.get_model(20, 100, 512)
    model.summary()
