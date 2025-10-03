#!/bin/bash

echo "Starting download the movie dataset from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata"
mkdir -p ./data/raw
curl -L -o ./data/raw/tmdb-movie-metadata.zip\
  https://www.kaggle.com/api/v1/datasets/download/tmdb/tmdb-movie-metadata
#!/bin/bash
curl -L -o ./data/raw/the-movies-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/rounakbanik/the-movies-dataset

echo "Unzipping the dataset from path: ./data/tmdb-movie-metadata.zip to ./data"
unzip -o ./data/tmdb-movie-metadata.zip -d ./data
unzip -o ./data/the-movies-dataset.zip -d ./data

echo "Cleaning up the zip file"
rm ./data/tmdb-movie-metadata.zip
rm ./data/the-movies-dataset.zip

echo "Download and extraction completed."
