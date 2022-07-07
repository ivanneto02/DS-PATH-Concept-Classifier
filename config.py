# Data parameters
DATA_PATH = "D:\\Documents\\Research\\DSPATH\\data"
DATA_FILE = "scraped_preprocessed_fulltext_2.csv"
NROWS = None
FEATURES = 300
SEP_COLUMN = 7
TARGET_COLUMN = "concept_type"
CLASSES = 2

# Model parameters
TFIDF_MODEL_NAME = "TFIDFModel"

# must add up to 1
TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

# hyperparameters
EPOCHS = 20
BATCH_SIZE = 128