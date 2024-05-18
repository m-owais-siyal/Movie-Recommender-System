import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_content_profile(tags_df, movies_df):
    movies_df['tags'] = movies_df['movieId'].map(
        tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x))
    )
    movies_df['tags'] = movies_df['tags'].fillna('')

def compute_item_similarity(movies_df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tag_matrix = vectorizer.fit_transform(movies_df['tags'])
    return cosine_similarity(tag_matrix, tag_matrix)

def recommend_movies(content_similarity, movies_df, rated_movies, top_n=5):
    similar_scores = content_similarity.sum(axis=0)
    similar_movies_idx = similar_scores.argsort()[-top_n:][::-1]
    return movies_df.iloc[similar_movies_idx]['title'].tolist()

def content_based_filtering(userid):
    ratings_df = pd.read_csv('ratings.csv')
    movies_df = pd.read_csv('movies.csv')
    tags_df = pd.read_csv('tags.csv')
    
    userid = int(userid)
    if userid not in ratings_df['userId'].unique():
        return ["Invalid user ID."]
    
    build_content_profile(tags_df, movies_df)
    content_similarity = compute_item_similarity(movies_df)
    
    user_rated_movies = ratings_df[ratings_df['userId'] == userid]['movieId']
    recommended_movies = recommend_movies(content_similarity, movies_df, user_rated_movies)
    
    return recommended_movies
