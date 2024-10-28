import os
from lib.cnn.models.inception_env import InceptionEnv
from lib.cnn.train import fit
from lib.cnn.predict import predict
from lib.evaluation import evaluate
from lib.metrics import RPrecision, F1Score

from lib.data.patchproviders import JpegPatchProvider, GLC23AltiPatchProviderNPY, MultiPatchProvider
from lib.data.datasets import GLC23PatchesDataset, GLC23PatchesDatasetMultiLabel, MetaDataset

# SETTINGS
# observations
OCC_PATH_PO_GLC_train = './data/PO_glc23_train.csv'
OCC_PATH_PA_GLC_train = './data/PA_glc23_train.csv'
OCC_PATH_PA_GLC_test_GPN_train = './data/PA_glc23_test_GPN_train.csv'
OCC_PATH_PA_EVA_train = './data/PA_eva_glc2023_train.csv'

OCC_PATH_PA_GLC_test_GPN_test = './data/PA_glc23_test_GPN_test.csv'
OCC_PATH_PA_EVA_test = './data/PA_eva_glc2023_val.csv'

# XP path
OUTPUT_PATH = './output/'

# model params
DROPOUT = 0.5

# training
BATCH_SIZE = 512
ITERATIONS = [8, 12, 16, 20, 24]
LOG_MODULO = 1000
VAL_MODULO = 1
LR = 0.01
GAMMA = 0.1

# evaluation
METRICS = (RPrecision(), F1Score())

# READ DATASET

root_data = './data/'

p_rgbi_glc23 = JpegPatchProvider(root_data+'patches_rgbi_glc/')
p_rgbi_eva = JpegPatchProvider(root_data+'patches_rgbi_eva/')
p_alti_glc = GLC23AltiPatchProviderNPY(root_data+'patches_alti_glc/')
p_alti_eva = GLC23AltiPatchProviderNPY(root_data+'patches_alti_eva/')

provider_eva = MultiPatchProvider((p_rgbi_eva, p_alti_eva))
provider_glc = MultiPatchProvider((p_rgbi_glc23, p_alti_glc))

print('---data loaded---')

# constructing pytorch dataset
po_glc_train_set = GLC23PatchesDataset(OCC_PATH_PO_GLC_train, (provider_glc,), nb_labels=10040)
pa_glc_train_set = GLC23PatchesDataset(OCC_PATH_PA_GLC_train, (provider_glc,), nb_labels=10040)
pa_glc_test_GPN_train_set = GLC23PatchesDataset(OCC_PATH_PA_GLC_test_GPN_train, (provider_glc,), nb_labels=10040)
pa_eva_train_set = GLC23PatchesDataset(OCC_PATH_PA_EVA_train, (provider_eva,), nb_labels=10040, id_name='eva_id')

train_set = MetaDataset((po_glc_train_set, pa_eva_train_set, pa_glc_test_GPN_train_set, pa_eva_train_set)) # concatenate all occs in 1 dataset

val_set_glc = GLC23PatchesDatasetMultiLabel(OCC_PATH_PA_GLC_test_GPN_test, (provider_glc,), nb_labels=10040)
val_set_eva = GLC23PatchesDatasetMultiLabel(OCC_PATH_PA_EVA_test, (provider_eva,), nb_labels=10040, id_name='eva_id')

validation_set = MetaDataset((val_set_glc, val_set_eva))

print('---dataset created---')

# CONSTRUCT MODEL
model = InceptionEnv(dropout=DROPOUT, n_labels=10040, n_input=5, kernel_size=15)

print('---model loaded--- (start learning)')

# FITTING THE MODEL
fit(
    model,
    train=train_set, validation=validation_set,
    batch_size=BATCH_SIZE, iterations=ITERATIONS, log_modulo=LOG_MODULO, val_modulo=VAL_MODULO, lr=LR, gamma=GAMMA,
    metrics=METRICS, output_path=OUTPUT_PATH, n_workers=24
)


# FINAL EVALUATION ON TEST SET
predictions, labels = predict(model, validation_set)
print(evaluate(predictions, labels, METRICS, final=True))

print('---finished---')