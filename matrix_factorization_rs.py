from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import pandas as pd

def matrix_factorization_rs(user_id):
    # Load ratings and movies data
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    tags = pd.read_csv('tags.csv')

    # Convert the ratings data to Surprise's Dataset format
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)

    # Initialize the SVD algorithm
    algo = SVD()
    algo.fit(trainset)

    # Function to recommend movies based on user ID
    def recommend_movies(user_id, n_top=5):
        user_id = int(user_id)
        all_movie_ids = ratings['movieId'].unique()
        rated_movies = ratings[ratings['userId'] == user_id]['movieId']
        unrated_movies = [mid for mid in all_movie_ids if mid not in rated_movies]

        predictions = [algo.predict(user_id, mid) for mid in unrated_movies]
        top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)[:n_top]

        top_movie_ids = [pred.iid for pred in top_predictions]
        recommended_movies = movies[movies['movieId'].isin(top_movie_ids)][['movieId', 'title']]

        # Drop duplicates based on the movie title to avoid duplicates
        recommended_movies = recommended_movies.drop_duplicates(subset='title')

        return recommended_movies['title'].tolist()

    return recommend_movies(user_id, n_top=5)
