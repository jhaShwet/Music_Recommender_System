import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Flatten, Dense
from keras.optimizers import Adam

def recommend_songs_for_user(user_id, data_path, embedding_dim=50, epochs=100, batch_size=32, N=10):
    # Load and preprocess the data
    data = pd.read_csv(data_path)
    
    user_ids = data['Position'].unique()
    song_ids = data['Song Name'].unique()

    n_users = len(user_ids)
    n_songs = len(song_ids)

    user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    song_id_to_index = {song_id: index for index, song_id in enumerate(song_ids)}

    data['User_Index'] = data['Position'].map(user_id_to_index)
    data['Song_Index'] = data['Song Name'].map(song_id_to_index)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Build the deep learning model
    user_id_input = Input(shape=(1,))
    song_id_input = Input(shape=(1,))

    user_embedding = Embedding(input_dim=n_users, output_dim=embedding_dim)(user_id_input)
    song_embedding = Embedding(input_dim=n_songs, output_dim=embedding_dim)(song_id_input)

    user_embedding_flat = Flatten()(user_embedding)
    song_embedding_flat = Flatten()(song_embedding)

    dot_product = Dot(axes=1)([user_embedding_flat, song_embedding_flat])

    output = Dense(1)(dot_product)

    model = Model(inputs=[user_id_input, song_id_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train the model
    model.fit([train_data['User_Index'], train_data['Song_Index']], train_data['Days'], epochs=epochs, batch_size=batch_size)

    # Get recommendations
    user_index = user_id_to_index[user_id]
    all_song_indices = np.arange(n_songs)
    unrated_song_indices = np.setdiff1d(all_song_indices, train_data[train_data['Position'] == user_id]['Song_Index'].values)
    user_ratings = model.predict([np.full_like(unrated_song_indices, user_index), unrated_song_indices])
    top_song_indices = unrated_song_indices[np.argsort(-user_ratings.flatten())[:N]]
    top_song_ids = [song_ids[i] for i in top_song_indices]

    return top_song_ids
# Replace 'path_to_dataset.csv' with the actual path to your dataset
data_path = 'Spotify_final_dataset.csv'
user_id = 1
recommended_songs = recommend_songs_for_user(user_id, data_path)

print("Top recommended songs:")
for i, song_id in enumerate(recommended_songs, 1):
    print(f"{i}. {song_id}")
