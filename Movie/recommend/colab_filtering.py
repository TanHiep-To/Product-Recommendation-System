import os
import sys
import pandas as pd
from surprise import SVD, Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ColabFiltering:
    def __init__(self, file_path="./data/raw/ratings.csv") -> None:
        """Initialize with the path to the CSV file containing movie data."""
        self.data = pd.read_csv(file_path)
        print("Data Loaded:\n", self.data.head())  # Check if data is loaded
        self.reader = Reader(rating_scale=(1, 5))
        self.dataset = Dataset.load_from_df(self.data[['userId', 'movieId', 'rating']], self.reader)
        self.model = SVD()

    def train_model(self) -> None:
        """Train the SVD model on the dataset."""
        if self.model is None or self.dataset is None:
            raise ValueError("Model and dataset must be initialized before training.")
        print("Training model...")
        trainset = self.dataset.build_full_trainset()
        self.model.fit(trainset)
        print("Model training completed.")

    def evaluate_model(self) -> None:
        """Evaluate the model using cross-validation."""
        if self.model is None or self.dataset is None:
            raise ValueError("Model and dataset must be initialized before evaluation.")
        results = cross_validate(self.model, self.dataset, measures=['RMSE', 'MAE'])
        print(f"Evaluation results: {results}")

    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """Recommend top N movies for a given user ID."""
        if self.model is None or self.dataset is None:
            raise ValueError("Model and dataset must be initialized before making recommendations.")

        print(f"Recommending movies for User ID: {user_id}")

        # Get a list of all movie IDs
        all_movie_ids = self.data['movieId'].unique()

        # Get the list of movie IDs the user has already rated
        rated_movie_ids = self.data[self.data['userId'] == user_id]['movieId'].unique()

        # Filter out the movies the user has already rated
        movie_ids_to_predict = [mid for mid in all_movie_ids if mid not in rated_movie_ids]

        # Predict ratings for the movies the user hasn't rated yet
        predictions = [self.model.predict(user_id, mid) for mid in movie_ids_to_predict]
        for prediction in predictions[:5]:  # Print first 5 predictions to check
            print(f"Predicted rating for movie {prediction.iid}: {prediction.est}")

        # Sort predictions by estimated rating in descending order
        predictions.sort(key=lambda x: x.est, reverse=True)

        # Get the top N movie IDs
        top_movie_ids = [pred.iid for pred in predictions[:top_n]]

        # Return the recommended movies as a DataFrame
        recommended_movies = self.data[self.data['movieId'].isin(top_movie_ids)].drop_duplicates('movieId')
        return recommended_movies[['movieId', 'title']].reset_index(drop=True)


def main() -> None:
    """Main function to demonstrate the recommendation system."""
    file_path = './data/raw/ratings.csv'
    test_subset = pd.read_csv(file_path).head(100)
    print(f"Test Subset:\n {test_subset.head()}")  # Check the first few rows
    unique_user_ids = test_subset['userId'].unique()
    print(f"Unique User IDs: {unique_user_ids}")  # Check unique user IDs

    colab_filtering = ColabFiltering(file_path)
    colab_filtering.train_model()
    colab_filtering.evaluate_model()

    # Recommend movies for the first unique user ID
    for user_id in unique_user_ids[:1]:
        recommendations = colab_filtering.recommend(user_id, top_n=10)
        print(f"Top 10 recommendations for User ID {user_id}:\n", recommendations)

if __name__ == "__main__":
    main()
