import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from data.process import DataProcessor

def load_and_process_data(file_path="./data/ratings_Beauty.csv"):
    processor = DataProcessor(file_path)
    processor.process_data()
    return processor

class BeautyRecommender:
    def __init__(self, n_components=10):
        self.processor = load_and_process_data()
        self.data = self.processor.get_data()
        self.n_components = n_components
        self._prepare_sparse_matrix()

    def _prepare_sparse_matrix(self):
        self.data['user_index'] = self.data['UserId'].astype('category').cat.codes
        self.data['product_index'] = self.data['ProductId'].astype('category').cat.codes

        # Create a sparse matrix
        self.ratings_matrix = csr_matrix((
            self.data['Rating'],
            (self.data['user_index'], self.data['product_index'])
        ))

        self.product_id_map = dict(enumerate(self.data['ProductId'].astype('category').cat.categories))
        self.product_index_map = {v: k for k, v in self.product_id_map.items()}

    def recommend(self, product_id, num_recommendations=5):
        if product_id not in self.product_index_map:
            raise ValueError(f"Product ID {product_id} not found in the dataset.")

        svd = TruncatedSVD(n_components=self.n_components)
        latent_matrix = svd.fit_transform(self.ratings_matrix.T)

        product_index = self.product_index_map[product_id]
        product_vector = latent_matrix[product_index].reshape(1, -1)

        similarities = cosine_similarity(product_vector, latent_matrix).flatten()
        similar_indices = similarities.argsort()[-num_recommendations-1:-1][::-1]

        similar_products = [self.product_id_map[i] for i in similar_indices]
        return similar_products

if __name__ == "__main__":
    recommender = BeautyRecommender()
    test_product_ids = recommender.processor.subset_data['ProductId'].unique()

    for product_id in test_product_ids[:5]:  # Test with first 5 unique product IDs
        recommendations = recommender.recommend(product_id)
        print(f"Recommendations for Product ID {product_id}: {recommendations}")