#!/bin/bash
DATA_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Downloading MNIST dataset into the data/ directory..."
curl -sS -L -o ${DATA_DIR}/mnist-dataset.zip\
   https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset

unzip -qq ${DATA_DIR}/mnist-dataset.zip && rm ${DATA_DIR}/mnist-dataset.zip && echo "Done!"
