import pandas as pd

class DataProcessor:
    def __init__(self, file_path = "./data/ratings_Beauty.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.subset_data = self.data.sample(frac=0.15, random_state=42)  # Using 15% of data for processing
    
    def get_data(self):
        return self.data
    
    def process_data(self):
        # Example processing: Remove duplicates and handle missing values
        self.data.drop_duplicates(inplace=True)
        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        
        # Summary statistics
        print("Data Summary:")
        print(self.data.describe())
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        print("\nUnique Values:")
        print(self.data.nunique())
        return self

if __name__ == "__main__":
    processor = DataProcessor()
    data = processor.process_data()
    print(data.head())