# SMS Spam Detector 📩

A binary text classification system that detects spam SMS messages using a Bidirectional LSTM neural network built with TensorFlow/Keras.

## Project Structure

```
sms-spam-detector/
├── data/
│   └── spam.csv              # UCI SMS Spam Collection dataset
├── artifacts/
│   ├── best_model.keras      # Best checkpoint (saved by EarlyStopping)
│   └── tokenizer.pkl         # Fitted tokenizer
├── logs/
│   └── training_log.csv      # Per-epoch metrics
├── tests/
│   └── test_data.py          # Unit & integration tests
├── preprocess.py             # Text cleaning, tokenization, splitting
├── train.py                  # Model definition & training pipeline
├── evaluate.py               # Metrics, confusion matrix, error analysis
├── requirements.txt
└── environment.yml
```

## Dataset

**UCI SMS Spam Collection** — 5,572 English SMS messages (4,825 ham / 747 spam).  
Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection  
License: Creative Commons Attribution 4.0 (CC BY 4.0)

## Setup

### Option A — pip
```bash
pip install -r requirements.txt
```

### Option B — conda
```bash
conda env create -f environment.yml
conda activate sms-spam-detector
```

## Usage

### 1. Download dataset
Place `spam.csv` inside the `data/` folder.

### 2. Train
```bash
python train.py --epochs 10 --batch_size 32 --lr 0.001 --embed_dim 64 --dropout 0.4
```

### 3. Evaluate
```bash
python evaluate.py
```

### 4. Run tests
```bash
pytest tests/ -v
```

## Model Architecture

```
Embedding(10000, 64) → BiLSTM(64) → GlobalMaxPool → Dense(64, ReLU) → Dropout(0.4) → Dense(1, Sigmoid)
```

**Loss:** Binary Cross-Entropy  
**Optimizer:** Adam (lr=0.001)  
**Primary metric:** F1-score (spam class)

## Results

| Split | Accuracy | F1 (spam) | AUROC |
|-------|----------|-----------|-------|
| Val   | ~98.5%   | ~0.96     | ~0.99 |
| Test  | ~98.3%   | ~0.95     | ~0.99 |

## Reproducibility

Seeds set in: `os`, `random`, `numpy`, `tensorflow`.  
Remaining nondeterminism: GPU-level float accumulation order in CuDNN LSTM kernels.

## Author

**abdullahsaim** — Innovate Technologies  
[github.com/abdullahsaim](https://github.com/abdullahsaim)
