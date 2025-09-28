"""Demographic filtering for movie recommendations."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pre_processing import MovieDataset

class GraphicFiltering:
    """Class implementing demographic filtering for movie recommendations."""

    def __init__(self, data):
        """Initialize with movie data."""
        self.data = data

    def weighted_rating(self, row):
        """Calculate weighted rating using IMDB formula."""
        mean_vote = self.data['vote_average'].mean()
        min_votes = self.data['vote_count'].quantile(0.90)
        votes = row['vote_count']
        rating = row['vote_average']
        return (votes / (votes + min_votes) * rating) + (min_votes / (min_votes + votes) * mean_vote)

    def recommend(self, top_n=10):
        """Recommend top N movies based on weighted rating."""
        movies = self.data.copy()
        movies['score'] = movies.apply(self.weighted_rating, axis=1)
        # Sort movies based on score
        movies = movies.sort_values('score', ascending=False)
        # Return top N movies
        return movies.head(top_n)

# Example usage:
if __name__ == "__main__":
    dataset = MovieDataset()
    dataset.load_data()
    dataset.preprocess_data()
    processed_data = dataset.data
    graphic_filter = GraphicFiltering(processed_data)
    recommendations = graphic_filter.recommend(top_n=10)
    print("Top 10 movie recommendations based on graphic filtering:")
    print(recommendations[['title', 'vote_count', 'vote_average', 'score']])