# GeoPlantNet model training from 2023

On jeanzay codes and models are in `/gpfsdswork/projects/rech/igz/commun/GeoPlantNet_Models/`, models and all output files related to their learning process are in the folders `models/`

## Setup

first, if on jeanzay:
`module load python/3.11.5` (do a `module purge` before if some modules are alredy loaded)

create a conda env from the yml in `lib`:\
```conda env create -f gpn_env.yml```

unzip `Deep-SDM.zip`

install the lib in the conda env:\
`cd Deep-SDM`\
`conda activate gpn`\
`pip install --upgrade .`


## occurrences/plots files

observations are split in 7 files:
- `PO_glc23_train`: the presence only observation of glc23 (removing 5000 occs for validations)
- `PO_glc23_val`: 5000 precense only occurrences from glc23 for validation
- `PA_glc23_train`: the public presence-absence from train/val of glc23
- `PA_glc23_test_GPN_train`: 95% of the presence-absence from the private test set of glc23 for training GPN models
- `PA_glc23_test_GPN_test`: 5% of the presence-absence from the private test set of glc23 for validating GPN models
- `PA_eva_glc23_train`: 95% of 2023 eva extraction of presence-absence for the train of GPN models
- `PA_eva_glc23_test`: 5% of 2023 eva extraction of presence-absence for the validation of GPN models

**important note**: the ids of eva and glc observations are not compatible and overlaps.