echo "Starting download data from Kaggle"

mkdir -p ./data
# Download dataset from Kaggle
curl -L -o ./data/amazon-ratings.zip \
  https://www.kaggle.com/api/v1/datasets/download/skillsmuggler/amazon-ratings
echo "Data downloaded successfully."

# Unzip the downloaded file
echo "Unzipping the downloaded data..."
unzip -o ./data/amazon-ratings.zip -d ./data/
echo "Data unzipped successfully."
echo "Data download and extraction completed."

# Remove the zip file to save space
rm ./data/amazon-ratings.zip
echo "Cleaned up the zip file."