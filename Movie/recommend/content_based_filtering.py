import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sklearn.metrics.pairwise
from sklearn.feature_extraction.text import TfidfVectorizer
from pre_processing import MovieDataset

class ContentBasedFiltering:
    def __init__(self, data):
        self.data = data
        self.tfidf_matrix = self.calculate_tfidf()
        self.cosine_sim = self.cosine_similarity(self.tfidf_matrix)

    def cosine_similarity(self, matrix):
        cosine_sim = sklearn.metrics.pairwise.cosine_similarity(matrix, matrix)
        print(f"Cosine similarity matrix shape: {cosine_sim.shape}")
        return cosine_sim

    def calculate_tfidf(self):
        # Example: Using 'overview' column for TF-IDF vectorization
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.data['overview'])
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"Feature names: {tfidf.get_feature_names_out()[:10]}")  # Display first 10 feature names
        return tfidf_matrix
    
    def recommend(self, movie_title, top_n=10):
        # Placeholder for content-based filtering logic
        # In a real scenario, this would involve calculating similarity scores
        # based on movie features like genres, cast, etc.
        if movie_title not in self.data['title'].values:
            return f"Movie '{movie_title}' not found in the dataset."
        
        idx = self.data.index[self.data['title'] == movie_title][0]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        sim_scores = sim_scores[1:top_n+1] # Exclude the first one as it is the movie itself

        recommend_movie_indices = [i[0] for i in sim_scores]

        movies = self.data.iloc[recommend_movie_indices]
        return movies
    
if __name__ == "__main__":
    dataset = MovieDataset()
    dataset.load_data()
    dataset.preprocess_data()
    processed_data = dataset.data
    content_filter = ContentBasedFiltering(processed_data)
    recommendations = content_filter.recommend(movie_title="The Dark Knight Rises", top_n=10)
    print("Top 10 movie recommendations based on content filtering:")
    print(recommendations[['title', 'overview', 'vote_average', 'vote_count']])
