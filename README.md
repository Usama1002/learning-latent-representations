# Learning High-Quality Latent Representations for Anomaly Detection and Signal Integrity Enhancement in High-Speed Signals

This repository contains the official code for the paper:

**"Learning High-Quality Latent Representations for Anomaly Detection and Signal Integrity Enhancement in High-Speed Signals"**

This repository provides a complete pipeline for anomaly detection and signal integrity (SI) enhancement in DRAM signals using deep learning. The workflow includes data preparation, labeling, model training, latent vector extraction, and SI enhancement.

## Directory Structure

```
.
├── code/
│   ├── interp_data.py
│   ├── prepare_training_data.py
│   ├── eye_label.py
│   ├── dataloader.py
│   ├── train_latent_vectors.py
│   ├── extract_latent_vectors.py
│   ├── si_enhancement.py
│   ├── si_enhance_dataset.py
│   └── README.md
├── data/
│   ├── original_data/         # Raw CSV data files
│   ├── interp_data/           # Interpolated CSVs (optional)
│   └── prepared_data/         # Processed .npz files for training
├── models/
│   ├── autoencoder_classifier.pth
│   └── latent_vectors_train.npz
└── ...
```

## Prerequisites

- Python 3.8+
- PyTorch (>=1.10)
- numpy
- pandas
- scipy
- tqdm
- matplotlib

Install dependencies with:

```bash
pip install torch numpy pandas scipy tqdm matplotlib
```

## Step-by-Step Usage

### 0. Download and Extract Data
Download the `data.zip` file from [this link](https://drive.proton.me/urls/1QJJTY5604#igJCFtueM1g1) and extract its contents into the `data/original_data` folder:

```bash
# Download data.zip from the provided link
# Then extract it:
unzip data.zip -d data/original_data
```

### 1.  Interpolate Raw Data
If your raw CSV files have irregular time steps, interpolate them to uniform sampling:

```bash
python interp_data.py
```
- Input: `data/original_data/1_ISI.csv` (edit script for other files)
- Output: `data/interp_data/1_ISI.csv`
... and similar for other files.
### 2. Prepare Training Data and Labeling
Segment the signals, label them using eye diagram analysis, and save as `.npz` files:

```bash
python prepare_training_data.py
```
- Input: `data/original_data/*.csv`
- Output: `data/prepared_data/<file>/<valid|invalid>/chXX_segXXXX.npz`

#### How Labeling Works
- Uses `eye_label.py` to assign 'valid' or 'invalid' labels based on eye diagram criteria.
- You can run `eye_label.py` directly to test labeling on a single sample:

```bash
python eye_label.py
```

### 3. Data Loading Utility
- `dataloader.py` provides PyTorch Dataset and DataLoader utilities for the prepared `.npz` files.
- Used internally by training and evaluation scripts.

### 4. Train the Autoencoder-Classifier Model
Train the model on the prepared data:

```bash
python train_latent_vectors.py
```
- Output: `models/autoencoder_classifier.pth`

### 5. Extract Latent Vectors
After training, extract latent vectors for all training samples:

```bash
python extract_latent_vectors.py
```
- Output: `models/latent_vectors_train.npz` (contains latent vectors and labels)

### 6. Signal Integrity Enhancement
Enhance signals using the SI enhancement algorithm:

#### Option 1: Standalone Enhancement Script
```bash
python si_enhancement.py
```
- Output: `data/enhanced_data.npz`

## File Descriptions (in order of use)

- **interp_data.py**: Interpolates raw CSV data to uniform time steps (optional).
- **prepare_training_data.py**: Segments and labels data, saves as `.npz` files for training.
- **eye_label.py**: Contains the eye diagram labeling function; can be run standalone for testing.
- **dataloader.py**: PyTorch Dataset/DataLoader for loading `.npz` waveform data.
- **train_latent_vectors.py**: Defines and trains the autoencoder-classifier model.
- **extract_latent_vectors.py**: Extracts latent vectors from the trained encoder for all training data.
- **si_enhancement.py**: Implements and applies the SI enhancement algorithm to all signals.
