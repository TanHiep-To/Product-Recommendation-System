"""Beauty product recommendation system using collaborative filtering."""

from typing import List

from data.process import DataProcessor
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def load_and_process_data(
    file_path: str = "./data/ratings_Beauty.csv",
) -> DataProcessor:
    """Load and process beauty product ratings data."""
    processor = DataProcessor(file_path)
    processor.process_data()
    return processor


class BeautyRecommender:
    """Recommender system for beauty products using collaborative filtering."""

    def __init__(self, n_components: int = 10) -> None:
        """Initialize the recommender with the number of components for SVD."""
        self.processor = load_and_process_data()
        self.data = self.processor.get_data()
        self.n_components = n_components
        self.ratings_matrix = None
        self.product_id_map = {}
        self.product_index_map = {}
        self._prepare_sparse_matrix()

    def _prepare_sparse_matrix(self):
        """Prepare the sparse matrix from the data for SVD."""
        self.data['user_index'] = (
            self.data['UserId'].astype('category').cat.codes
        )
        self.data['product_index'] = (
            self.data['ProductId'].astype('category').cat.codes
        )

        # Create a sparse matrix
        self.ratings_matrix = csr_matrix(
            (
                self.data['Rating'],
                (self.data['user_index'], self.data['product_index']),
            )
        )

        self.product_id_map = dict(
            enumerate(self.data['ProductId'].astype('category').cat.categories)
        )
        self.product_index_map = {v: k for k, v in self.product_id_map.items()}

    def recommend(
        self, item_id: str, num_recommendations: int = 5
    ) -> List[str]:
        """Get recommendations for a given product."""
        if item_id not in self.product_index_map:
            raise ValueError(f"Product ID {item_id} not found in the dataset.")

        svd = TruncatedSVD(n_components=self.n_components)
        latent_matrix = svd.fit_transform(self.ratings_matrix.T)

        product_index = self.product_index_map[item_id]
        product_vector = latent_matrix[product_index].reshape(1, -1)

        similarities = cosine_similarity(
            product_vector, latent_matrix
        ).flatten()
        similar_indices = similarities.argsort()[
            -num_recommendations - 1 : -1
        ][::-1]

        similar_products = [self.product_id_map[i] for i in similar_indices]
        return similar_products

    def get_popular_products(self, n: int = 10) -> List[str]:
        """Get the most popular products based on rating count."""
        product_ratings = self.data.groupby('ProductId')['Rating'].agg(
            ['count', 'mean']
        )
        popular = product_ratings.sort_values('count', ascending=False)
        return popular.head(n).index.tolist()

    def get_similar_users(self, user_id: str, n: int = 5) -> List[str]:
        """Find users with similar rating patterns."""
        if user_id not in self.data['UserId'].unique():
            raise ValueError(f"User ID {user_id} not found")
        user_ratings = self.data.pivot_table(
            index='UserId', columns='ProductId', values='Rating'
        )
        user_sim = cosine_similarity(user_ratings.fillna(0))
        user_idx = user_ratings.index.get_loc(user_id)
        similar_scores = list(enumerate(user_sim[user_idx]))
        similar_users = sorted(
            similar_scores, key=lambda x: x[1], reverse=True
        )[1 : n + 1]
        return [user_ratings.index[idx] for idx, _ in similar_users]


if __name__ == "__main__":
    recommender = BeautyRecommender()
    test_product_ids = recommender.processor.subset_data['ProductId'].unique()

    for product_id in test_product_ids[
        :5
    ]:  # Test with first 5 unique product IDs
        recommendations = recommender.recommend(product_id)
        print(
            f"Recommendations for Product ID {product_id}: {recommendations}"
        )
