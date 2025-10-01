"""Demographic filtering for movie recommendations."""

import os
import sys
from Movie.pre_processing import MovieDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        first_part = (votes / (votes + min_votes)) * rating
        second_part = (min_votes / (min_votes + votes)) * mean_vote

        return first_part + second_part

    def recommend(self, top_n=10):
        """Recommend top N movies based on weighted rating."""
        movies = self.data.copy()
        movies['score'] = movies.apply(self.weighted_rating, axis=1)
        # Sort movies based on score
        movies = movies.sort_values('score', ascending=False)
        # Return top N movies
        return movies.head(top_n)

def main() -> None:
    """Main function to demonstrate filtering."""
    dataset = MovieDataset()
    dataset.load_data()
    dataset.preprocess_data()
    data = dataset.data
    graphic_filter = GraphicFiltering(data)
    recommendations = graphic_filter.recommend()
    print("Top movie recommendations:")
    print(recommendations[['title', 'vote_count', 'vote_average', 'score']])

# Example usage:
if __name__ == "__main__":
    main()
