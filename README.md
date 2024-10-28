# garden SDM v1 models training

## Setup

create a conda env from the yml in `lib`:\
```conda env create -f gpn_env.yml```

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
