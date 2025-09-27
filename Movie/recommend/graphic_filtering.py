# Graphic Filtering
# This module implements graphic filtering techniques for movie recommendations.
# We need a metric to score or rank movies based on their popularity and quality.
# - Calculate a weighted rating score for each movie using the IMDB formula.
# - Sort the movies based on the weighted rating score.
# - Select the top N movies to recommend.
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pre_processing import MovieDataset

class GraphicFiltering:
    def __init__(self, data):
        self.data = data
    
    def weighted_rating(self, row):
        C = self.data['vote_average'].mean()
        m = self.data['vote_count'].quantile(0.90)

        v = row['vote_count']
        R = row['vote_average']

        # Calculate the weighted rating
        # Formula: (v/(v+m) * R) + (m/(m+v) * C)
        # Where:
        # v = number of votes for the movie
        # m = minimum votes required to be listed in the chart
        # R = average rating of the movie
        # C = mean vote across the whole report

        return (v / (v + m) * R) + (m / (m + v) * C)

    def recommend(self, top_n=10):
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