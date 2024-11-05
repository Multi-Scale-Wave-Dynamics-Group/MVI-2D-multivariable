# Multi-variable Imputation 2D (MVI-2D): A Physics-Informed Multi-variable Transformer Model for Imputing Missing Solidum Lidar Observations over the Andes
This repository contains code and data fro a multi-variable imputation model designed for atmopsheric data measured by LiDARs. The model incorporates transfomer and convolutional neural networks to perform missing data imputation task

## Features
**Multivarible input**: Sodium density, wind and temperature data as inputs

**Rotary Embedding**: Ultilize the rotary embedding for altitude and time dimensions in spatial and temporal contexts

**Local Window Attention**: Implements a localized attention mechanism via convolutional neural network to focus on independent data regions

**Mask-guided attention**: Enhances imputation by focusing on missing data regions

**Loss Functions**: Includes 1) reconstruction loss, 2) imputation loss, 3) second derivative loss, and 4) physcis-based constraints

## Code package structure
MVI-2D-multivariable/

├── data/                    # Data storage

├── model/                   # Atmospheric data saving directory from model "msise00"

├── functions/               # Utility functions

├── transfomer/              # Training script

├── debug/                   # Testing script

├── Results/                 # Trained model weights and figures

└── README.md





