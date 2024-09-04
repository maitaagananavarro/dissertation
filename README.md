# Experiment Setup and Reproduction

## Source Code

The setup for running benchmark experiments is available on [ClimateSet’s GitHub repository](https://github.com/RolnickLab/ClimateSet/). Instructions and code for additional out-of-distribution experiments created in this work can also be found in this repository.

## Instructions

### 1. Setup Benchmark Experiments

1. **Download Dataset**: Follow ClimateSet’s instructions for [downloading their core dataset](https://github.com/RolnickLab/ClimateSet/blob/main/README.md#downloading-the-core-dataset).
2. **Setup Single Emulation Experiment**: Follow ClimateSet's guidelines to [set up a Single Emulation experiment](https://github.com/RolnickLab/ClimateSet/blob/main/README.md#downloading-the-core-dataset) using their dataset.
   - **Note**: ClimaX and ClimaX Frozen have different setup requirements compared to U-Net and Conv-LSTM.

### 2. Reproduce the Experiments in This Work

1. **Download and Run the Template Files**:
   - **Method #1**: `setup_timeshift.py`
   - **Method #2**: `setup_ssp.py`
   - **Method #3**: `setup_lowconfidence.py`

2. **For Method #3**:
   - Add the additional `methods_confidence.py` file to ClimateSet's `emulator` folder, which contains all necessary methods for post-testing based on confidence thresholds.
   - Import `methods_confidence.py` into ClimateSet’s existing `emulator/train.py` file and integrate the methods as described in the repository.

3. **Run the Template Files**:
   - Execute the template files as usual, following ClimateSet’s instructions.

### 3. Examples to Try

- **Method #1**: Run a U-Net model on an MPI-ESM1-2-HR dataset.
  ```bash
  python emulator/run.py experiment=single_emulator/unet/MPI-ESM1-2-HR/unet_experiment_timeshift.yaml
  
- **Method #2**: Run a Conv-LSTM model on an AWI-CM-1-1-MR dataset, tested on SSP3-7.0.
  ```bash
  python emulator/run.py experiment=single_emulator/convlstm/AWI-CM-1-1-MR/convlstm_experiment_ssp370.yaml

- **Method #3**: Run a ClimaX model on an FGOALS-f3-L dataset.
  ```bash
  python emulator/run.py experiment=single_emulator/climax/FGOALS-f3-L/climax_experiment_confidence.yaml


