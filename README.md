# "Do Current Climate Models Work in Changing Climate Dynamics?" - Experiment Setup and Reproduction

## Purpose

The purpose of these instructions is to guide users through the setup and reproduction of benchmark experiments for evaluating climate models in the context of changing climate dynamics, explored in the study "Do Current Climate Models Work in Changing Climate Dynamics?". Users will:
1. **Set Up Benchmark Experiments**: Configure and run benchmark experiments using ClimateSet’s core dataset and emulation tools.
2. **Reproduce Experiments**: Implement and execute specific methods for testing model robustness under various conditions, including different time periods, SSP scenarios, and confidence intervals.
3. **Validate Models**: Utilize the provided template files and methods to verify the performance of different climate models, ensuring they are robust to changing climate dynamics.

While this study focuses on ClimateSet’s setup and experiments, users interested in adapting the framework to evaluate other climate models are encouraged to do so. Although detailed instructions for this adaptation are not provided here, the framework is designed to be flexible and can be modified to accommodate different models and datasets.

## Source Code

The setup for running benchmark experiments is available on [ClimateSet’s GitHub repository](https://github.com/RolnickLab/ClimateSet/). Instructions and code for additional out-of-distribution experiments created in this work can also be found in this repository.

## Instructions

### 1. Setup Benchmark Experiments

1. **Download Dataset**: Follow ClimateSet’s instructions for [downloading their core dataset](https://github.com/RolnickLab/ClimateSet/blob/main/README.md#downloading-the-core-dataset).
2. **Setup Single Emulation Experiment**: Follow ClimateSet's guidelines to [set up a Single Emulation experiment](https://github.com/RolnickLab/ClimateSet/blob/main/README.md#downloading-the-core-dataset) using their dataset.
   - **Note**: ClimaX and ClimaX Frozen have different setup requirements compared to U-Net and Conv-LSTM.

### 2. Reproduce the Experiments in This Work

1. **Download and Run the Template Files:**
Running the files will create the templates corresponding to each method.
   - **Method #1 - Splitting Based on Time Period**: `setup_timeshift.py`
   - **Method #2 - Changing SSP Scenario**: `setup_ssp.py`
   - **Method #3 - Confidence-Interval Based**: `setup_confidence.py`

3. **For Method #3**:
   - Add the additional `methods_confidence.py` file to ClimateSet's `emulator` folder, which contains all necessary methods for post-testing based on confidence thresholds.
   - Import `methods_confidence.py` into ClimateSet’s existing `emulator/train.py` file and integrate the methods as described in the repository. See `train.py` in this repository as an example of how to integrate the methods. 

4. **Run the Template Files**:
   - Execute the template files as usual, following ClimateSet’s instructions.

### 3. Examples to Try

- **Method #1**: Run a U-Net model on an MPI-ESM1-2-HR dataset, with different train and test time periods.
  ```bash
  python emulator/run.py experiment=single_emulator/unet/MPI-ESM1-2-HR/unet_experiment_timeshift.yaml
  
- **Method #2**: Run a Conv-LSTM model on an AWI-CM-1-1-MR dataset, tested on SSP3-7.0.
  ```bash
  python emulator/run.py experiment=single_emulator/convlstm/AWI-CM-1-1-MR/convlstm_experiment_ssp370.yaml

- **Method #3**: Run a ClimaX model on an FGOALS-f3-L dataset, and retrieve metrics for low confidence and confident points.
  ```bash
  python emulator/run.py experiment=single_emulator/climax/FGOALS-f3-L/climax_experiment_confidence.yaml


