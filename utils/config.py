from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models_file'
EMBEDDINGS_PATH = BASE_DIR / 'embeddings' / 'embeddings.pkl'
MY_EMBEDDINGS_PATH = BASE_DIR / 'embeddings' / 'my_embeddings.pkl'

MY_DATASET_PATH = DATA_DIR / 'train_data'
DATASET_PATH = DATA_DIR / 'lfw-deepfunneled'
MODEL_PATH = MODEL_DIR / 'resnet34_MXNET_E_SGD_REG_1e3_on_batch_true_lr1e1_random0_arc_S32_E1_BS512_casia_basic_agedb_30_epoch_36_0.949500.h5'
TEST_PATH = DATA_DIR / 'test_data'