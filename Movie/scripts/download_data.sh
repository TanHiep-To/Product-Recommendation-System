#!/bin/bash

echo "Starting download the movie dataset from https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata"

curl -L -o ./data/tmdb-movie-metadata.zip\
  https://www.kaggle.com/api/v1/datasets/download/tmdb/tmdb-movie-metadata

echo "Unzipping the dataset from path: ./data/tmdb-movie-metadata.zip to ./data"
unzip -o ./data/tmdb-movie-metadata.zip -d ./data

echo "Cleaning up the zip file"
rm ./data/tmdb-movie-metadata.zip

echo "Download and extraction completed."