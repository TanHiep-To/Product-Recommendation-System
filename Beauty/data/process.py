"""Data processing module for beauty product ratings."""

import pandas as pd

class DataProcessor:
    """Process and clean beauty product ratings data."""

    def __init__(self, file_path: str = "./data/ratings_Beauty.csv") -> None:
        """Initialize with data file path."""
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.subset_data = self.data.sample(frac=0.15, random_state=42)

    def get_data(self) -> pd.DataFrame:
        """Return the processed data."""
        return self.data

    def process_data(self) -> 'DataProcessor':
        """Process and clean the data."""
        self.data.drop_duplicates(inplace=True)
        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        return self

def main() -> None:
    """Main function for testing."""
    processor = DataProcessor()
    processor.process_data()
    print(processor.data.head())

if __name__ == "__main__":
    main()
