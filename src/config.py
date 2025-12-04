import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(BASE_DIR, "local_data", "csv", "sce_E3")
EMB_DIR = os.path.join(BASE_DIR, "local_data", "embeddings")

SEED = 42
NUM_CLASSES = 6
DIM_FEATURES = 512
BUFFER_SIZE = 42
AKD_LAMBDA = 1.0
KD_LAMBDA = 1.0

TASK_FILES = [
    ("2_4_train.csv", "2_4_val.csv", "2_4_test.csv"),
    ("0_5_train.csv", "0_5_val.csv", "0_5_test.csv"),
    ("3_1_train.csv", "3_1_val.csv", "3_1_test.csv")
]

PAPER_TARGETS = {
    'FT':   {'AACC': 0.3167, 'BWT': -0.8701, 'IM': -0.1942},
    'ER':   {'AACC': 0.5388, 'BWT': -0.3869, 'IM': -0.0942},
    'OURS': {'AACC': 0.5926, 'BWT': -0.4056, 'IM': -0.1604},
    'JT':   {'AACC': 0.7311, 'BWT': 0.0000,  'IM': 0.0000}
}
