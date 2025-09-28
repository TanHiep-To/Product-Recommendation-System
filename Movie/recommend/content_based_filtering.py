"""Content-based movie recommendation system."""

import os
import sys
from typing import Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics.pairwise
from pre_processing import MovieDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ContentBasedFiltering:
    """Class implementing content-based movie recommendations."""

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize with movie data."""
        self.data: pd.DataFrame = data
        self.tfidf_matrix = None
        self.cosine_sim = None
        self._initialize_matrices()
    
    def _initialize_matrices(self) -> None:
        """Initialize TF-IDF and cosine similarity matrices."""
        self.tfidf_matrix = self.calculate_tfidf()
        self.cosine_sim = self.cosine_similarity(self.tfidf_matrix)

    def cosine_similarity(self, matrix: Any) -> np.ndarray:
        """Compute cosine similarity matrix."""
        similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(matrix, matrix)
        print(f"Cosine similarity matrix shape: {similarity_matrix.shape}")
        return similarity_matrix

    def calculate_tfidf(self) -> Any:
        """Calculate TF-IDF matrix from movie overviews."""
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.data['overview'])
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"Feature names: {tfidf_vectorizer.get_feature_names_out()[:10]}")
        return tfidf_matrix
    
    def recommend(self, movie_title: str, top_n: int = 10) -> Union[str, pd.DataFrame]:
        """Recommend top N movies similar to the given movie title."""
        if movie_title not in self.data['title'].values:
            return f"Movie '{movie_title}' not found in the dataset."
        
        movie_idx = self.data.index[self.data['title'] == movie_title][0]
        similarity_scores = list(enumerate(self.cosine_sim[movie_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similar_movies = similarity_scores[1:top_n+1]
        movie_indices = [idx for idx, _ in similar_movies]
        
        return self.data.iloc[movie_indices]

def main() -> None:
    """Main function to demonstrate the recommendation system."""
    dataset = MovieDataset()
    dataset.load_data()
    dataset.preprocess_data()
    processed_data = dataset.data
    recommender = ContentBasedFiltering(processed_data)
    movie_title = "The Dark Knight Rises"
    recommendations = recommender.recommend(movie_title=movie_title, top_n=10)
    print(f"\nTop 10 movie recommendations similar to '{movie_title}':")
    print(recommendations[['title', 'overview', 'vote_average', 'vote_count']])

if __name__ == "__main__":
    main()
