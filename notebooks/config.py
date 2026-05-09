import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import shap

from pathlib import Path
from itertools import permutations, combinations

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(r'D:/thesis/iot-ids-ml-transferability-robustness')
RAW_DIR     = ROOT / 'data' / 'raw'
PROC_DIR    = ROOT / 'data' / 'processed'
SPLITS_DIR  = ROOT / 'data' / 'splits'
MODELS_DIR  = ROOT / 'models'
RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = ROOT / 'figures'

for d in [PROC_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATASETS     = ['bot-iot', 'unsw-nb15', 'ton-iot']
MODEL_NAMES  = ['RF', 'SVM', 'CNN', 'LSTM']
SEEDS        = [42, 101, 202, 303, 404]
RANDOM_STATE = 42
TEST_SIZE    = 0.20
MODEL_COLORS = {'RF': '#2196F3', 'SVM': '#4CAF50', 'CNN': '#FF9800', 'LSTM': '#9C27B0'}

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 11,
    'axes.titlesize': 12, 'font.family': 'DejaVu Serif'
})
