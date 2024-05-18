# Project Report: Recommender System

## Introduction

This project is a Recommender System that employs three different recommendation algorithms. The system is designed with a custom Tkinter GUI, allowing users to select a recommendation algorithm, input a user ID, and receive personalized movie recommendations. The three recommendation algorithms included are:

1. Content-Based Filtering
2. Matrix Factorization
3. Naive Recommender System

## Repository Structure

The GitHub repository contains the following files:

- **RsProject.py**: The main script to run the Tkinter GUI and manage user interactions.
- **content_based_rs.py**: Implements the Content-Based Filtering algorithm.
- **matrix_factorization_rs.py**: Implements the Matrix Factorization algorithm.
- **naive_rs.py**: Implements a Naive Recommender System.
- **movies.csv**: Dataset containing movie information.
- **ratings.csv**: Dataset containing user ratings for movies.
- **tags.csv**: Dataset containing tags associated with movies.

## Usage

### Prerequisites

Ensure you have Python installed on your system along with the necessary libraries:
- pandas
- numpy
- scikit-learn
- tkinter

You can install the required libraries using pip:
```sh
pip install pandas numpy scikit-learn tk
```

### Running the Recommender System

1. Clone the repository to your local machine:
    ```sh
    git clone https://github.com/m-owais-siyal/Movie-Recommender-System
    ```
   
2. Navigate to the project directory:
    ```sh
    cd repo
    ```

3. Run the main script:
    ```sh
    python RsProject.py
    ```

### Using the Tkinter GUI

Upon running `RsProject.py`, a Tkinter GUI will launch. Follow these steps to get recommendations:

1. **Select a Recommender Algorithm**: Use the drop-down menu to choose one of the three algorithms:
    - Content-Based Filtering
    - Matrix Factorization
    - Naive Recommender System

2. **Enter User ID**: Input the user ID for whom you want to generate recommendations.

3. **Get Recommendations**: Click the button to generate and display the recommendations for the specified user.

## Recommendation Algorithms

### Content-Based Filtering

Implemented in `content_based_rs.py`, this algorithm recommends movies based on the similarity of movie attributes (e.g., genre, tags) to the user's previously rated movies.

### Matrix Factorization

Implemented in `matrix_factorization_rs.py`, this algorithm uses matrix factorization techniques ,i.e. Singular Value Decomposition (SVD) to factorize the user-item interaction matrix and predict user ratings for unseen movies.

### Naive Recommender System

Implemented in `naive_rs.py`, this simple algorithm recommends the most popular movies or movies highly rated by similar users without considering specific user preferences deeply.

## Datasets

- **movies.csv**: Contains movie details such as movie IDs, titles, and genres.
- **ratings.csv**: Contains user ratings for movies, with columns for user ID, movie ID, rating, and timestamp.
- **tags.csv**: Contains user-assigned tags for movies, with columns for user ID, movie ID, tag, and timestamp.

## Conclusion

This project demonstrates a basic yet effective Recommender System using multiple algorithms and a user-friendly GUI. It provides a foundation for exploring more advanced recommendation techniques and improving user experience with personalized suggestions.

## Future Work

Potential improvements and extensions to this project include:

- Incorporating additional algorithms such as Collaborative Filtering.
- Enhancing the GUI for better user experience.
- Integrating more sophisticated data preprocessing techniques.
- Adding evaluation metrics to compare the performance of different algorithms.

---

For further details, refer to the individual script files and the comments within the code. Feel free to fork the repository and contribute to its development.
