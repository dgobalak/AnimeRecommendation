import pandas as pd
from fuzzywuzzy import process
import pickle

df_anime = pd.read_csv('anime.csv', usecols=['anime_id', 'name'], dtype={
                       'anime_id': 'int32', 'name': 'str'})
df_ratings = pd.read_csv('rating.csv', dtype={
                         'anime_id': 'int32', 'user_id': 'int32', 'rating': 'float32'})[['anime_id', 'user_id', 'rating']]

with open('ratings_model_knn.pkl', 'rb') as f:
    model_knn = pickle.load(f)

with open('ratings_matrix_knn.pkl', 'rb') as f:
    matrix = pickle.load(f)


def get_anime_id(name):
    anime_details = process.extractOne(name, df_anime['name'])
    # Ex: ('Naruto: Shippuden', 91, 615) gives the name, accuracy, and id number
    id = anime_details[2]
    return id


def get_anime_names(indices):
    names = [str(df_anime['name'][x]) for x in indices[0][1:]]
    return names


# Recommender(anime_name) => List of Recommended Anime
def recommender(anime_name, data_matrix, model, n_recommendations):
    id = get_anime_id(anime_name)
    print(f"Suggested anime if you\'ve watched {df_anime['name'][id]}")
    distances, indices = model.kneighbors(
        data_matrix[id], n_neighbors=n_recommendations)
    recommended = get_anime_names(indices)
    print("\n".join(recommended))


recommender('Fullmetal Alchemist: Brotherhood', matrix, model_knn, 20)
