'''
File name: config.py
Author: David Megli
Created: 2025-03-14
Description: hyperparameters for training
'''
# Hyperparameters
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 10

# Model parameters
INPUT_SIZE = 784   # 28x28 immagini MNIST
WIDTH = 128        # Dimensione dei layer nascosti
DEPTH = 2          # Numero di layer nascosti
OUTPUT_SIZE = 10   # Numero di classi (MNIST ha 10 cifre)

LAYER_SIZES = [INPUT_SIZE] + [WIDTH]*DEPTH + [OUTPUT_SIZE]