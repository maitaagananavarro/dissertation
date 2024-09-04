import os
import yaml
from string import Template

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def create_experiment_yaml(base_config_path, modifications, output_path):
    # Load the base configuration
    config = load_yaml(base_config_path)
    
    # Apply modifications
    for key, value in modifications.items():
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    
    # Save the modified configuration
    save_yaml(config, output_path)

# Base template YAML path (you can modify this)
base_template = """
defaults:
 - override /mode: exp.yaml
 - override /trainer: default.yaml
 - override /model: ${model}.yaml
 - override /callbacks: default.yaml
 - override /logger: wandb.yaml
 - override /datamodule: climate.yaml

name: "${train_model}_${model}_run-${run_id}"

seed: ${seed}

trainer:
 min_epochs: 1
 max_epochs: ${max_epochs}

model:
 loss_function: "climax_lon_lat_rmse"
 monitor: "val/llrmse_climax"
 finetune: False
 pretrained_run_id: null
 pretrained_ckpt_dir: null

datamodule:
  in_var_ids: ['BC_sum', 'CO2_sum', 'SO2_sum', 'CH4_sum']
  out_var_ids: ['tas', 'pr']
  train_historical_years: "${train_historical_years}"
  test_years: "${test_years}"
  train_models:  ["${train_model}"]
  seq_to_seq: True
  batch_size: 4
  channels_last: False
  eval_batch_size: 4

logger:
 wandb:
   tags: ["single_emulator", "${model}", "${train_model}", "tas+pr", "run${run_id}"]
"""

def create_experiments_timeshift():
    train_models = [
        'MPI-ESM1-2-HR',
        'AWI-CM-1-1-MR',
        'EC-Earth3',
        'FGOALS-f3-L',
        'BCC-CSM2-MR'
    ]
    
    for train_model in train_models:
        experiments = [
            {
                'model': 'unet',
                'train_model': train_model,
                'train_historical_years': '1850-2014',  
                'test_years':'2015-2023',              
                'max_epochs': 50,
                'seed': 22201,
                'run_id': 1,
                'output_dir': f'emulator/configs/experiment/single_emulator/unet/{train_model}'
            },
            {
                'model': 'convlstm',
                'train_model': train_model,
                'train_historical_years': '1850-2014',  
                'test_years':'2015-2023', 
                'max_epochs': 50,
                'seed': 22201,
                'run_id': 1,
                'output_dir': f'emulator/configs/experiment/single_emulator/convlstm/{train_model}'
            },
            {
                'model': 'climax',
                'train_model': train_model,
                'train_historical_years': '1850-2014',  
                'test_years':'2015-2023', 
                'max_epochs': 50,
                'seed': 22201,
                'run_id': 1,
                'output_dir': f'emulator/configs/experiment/single_emulator/climax/{train_model}'
            },
            {
                'model': 'climax_frozen',
                'train_model': train_model,
                'train_historical_years': '1850-2014',  
                'test_years':'2015-2023', 
                'max_epochs': 50,
                'seed': 22201,
                'run_id': 1,
                'output_dir': f'emulator/configs/experiment/single_emulator/climax_frozen/{train_model}'
            }
        ]

        for exp in experiments:
            template = Template(base_template)
            config_content = template.safe_substitute(
                model=exp['model'],
                train_model=exp['train_model'],
                train_historical_years=exp['train_historical_years'],
                test_years=exp['test_years'],
                max_epochs=exp['max_epochs'],
                seed=exp['seed'],
                run_id=exp['run_id']
            )
            
            output_path = os.path.join(exp['output_dir'], f'{exp["model"]}_experiment_timeshift.yaml')
            save_yaml(yaml.safe_load(config_content), output_path)
            print(f'Created {output_path}')

# Run the function to create the experiment YAML files
create_experiments_timeshift()
