"""Movie dataset preprocessing module."""

import ast
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings("ignore")


class MovieDataset:
    """Class for loading and preprocessing TMDB movie dataset."""

    def __init__(self, credits_file='data/raw/tmdb_5000_credits.csv', movies_file='data/raw/tmdb_5000_movies.csv'):
        """Initialize MovieDataset with file paths."""
        self.data = None
        self.credits_file = credits_file
        self.movies_file = movies_file

    def load_data(self):
        """Load movie data from CSV files and merge them."""
        print("Starting data loading...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        credits = pd.read_csv(os.path.join(base_dir, '.', self.credits_file), low_memory=False)
        movies = pd.read_csv(os.path.join(base_dir, '.', self.movies_file), low_memory=False)
        movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')
        movies.drop('movie_id', axis=1, inplace=True)
        movies = movies[['id', 'overview', 'genres', 'keywords', 'cast', 'crew', 'popularity',
                 'release_date', 'vote_average', 'vote_count', 'budget', 'revenue',
                 'runtime', 'status', 'tagline', 'title_x']]

        self.data = movies

    def convert_to_list(self, text):
        """Convert JSON-like string to list of names."""
        try:
            return [item['name'] for item in ast.literal_eval(text)]
        except (ValueError, SyntaxError):
            return []

    def extract_director(self, crew_list):
        """Extract director name from crew list."""
        try:
            crew = ast.literal_eval(crew_list)
            for member in crew:
                if member['job'] == 'Director':
                    return member['name']
        except (ValueError, SyntaxError):
            return np.nan
        return np.nan

    def preprocess_data(self):
        """Preprocess movie data by cleaning and transforming features."""
        if self.data is None:
            raise ValueError("Data not found. Please load the data first.")

        print("Starting data preprocessing...")
        movies = self.data
        # Convert JSON-like strings to lists
        movies['genres'] = movies['genres'].apply(self.convert_to_list)
        movies['keywords'] = movies['keywords'].apply(self.convert_to_list)
        movies['director'] = movies['crew'].apply(self.extract_director)
        movies.drop(columns=['crew'], inplace=True)

        # Handle missing values
        movies['overview'] = movies['overview'].fillna('')
        movies['tagline'] = movies['tagline'].fillna('')
        movies['cast'] = movies['cast'].fillna('[]').apply(self.convert_to_list)
        movies['popularity'] = movies['popularity'].fillna(movies['popularity'].median())
        movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
        movies['vote_average'] = movies['vote_average'].fillna(movies['vote_average'].median())
        movies['vote_count'] = movies['vote_count'].fillna(movies['vote_count'].median())
        movies['budget'] = movies['budget'].replace(0, np.nan).fillna(movies['budget'].median())
        movies['revenue'] = movies['revenue'].replace(0, np.nan).fillna(movies['revenue'].median())
        movies['runtime'] = movies['runtime'].fillna(movies['runtime'].median())

        # Remove rows with critical missing information
        movies.dropna(subset=['title_x', 'release_date'], inplace=True)
        movies.rename(columns={'title_x': 'title'}, inplace=True)
        movies.reset_index(drop=True, inplace=True)

        self.data = movies


if __name__ == "__main__":
    dataset = MovieDataset()
    dataset.load_data()
    dataset.preprocess_data()
    processed_data = dataset.data
    print("Data preprocessing completed. Here are the first few rows of the processed data:")
    print(processed_data.head())

    print("Summary of movies data:")
    print(processed_data.describe(include='all'))
