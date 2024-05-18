import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def naive_bayes_cf(user_input):
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    merged_data = pd.merge(ratings, movies, on='movieId')

    # Improved data preprocessing
    merged_data['title'] = merged_data['title'].str.lower()
    merged_data['genres'] = merged_data['genres'].str.lower()
    merged_data['text_features'] = merged_data['title'] + ' ' + merged_data['genres']

    # Adjust discretization strategy
    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    merged_data['rating_bin'] = discretizer.fit_transform(
        merged_data['rating'].values.reshape(-1, 1))

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(merged_data['text_features'])
    y = merged_data['rating_bin']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Adjust hyperparameters for MultinomialNB
    naive_bayes = MultinomialNB(alpha=0.5)
    naive_bayes.fit(X_train, y_train)

    predicted = naive_bayes.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)

    def recommend_movies(user_input, top_n=5):
        user_input_vector = vectorizer.transform([user_input])
        predicted_rating_bins = naive_bayes.predict(user_input_vector)
        top_movies = merged_data.loc[merged_data['rating_bin'].isin(predicted_rating_bins)].groupby(
            'movieId')['rating'].mean().reset_index().sort_values('rating', ascending=False).head(top_n)
        recommended_movies = pd.merge(top_movies, movies, on='movieId')
        return recommended_movies[['title', 'rating']]

    recommended_movies = recommend_movies(user_input, top_n=5)
    return recommended_movies['title'].tolist(), accuracy
