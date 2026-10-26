"""Run inference for all models defined in train_all.py with configuration defined in 
configs/inference.yml"""

from pathlib import Path
import train_all
from train_all import CommandRunner

import yaml

models = train_all.models

base_config_path = "configs/inference.yml"
# We create a new config file for each model to avoid overwriting the base config
config_path = "configs/inference_model.yaml"
command = ["python", "-u", "main.py", "--config_path", config_path]

command_runner = CommandRunner(stream_realtime=True, log_filepath="inference_all.log")
for model in models:

    with open(base_config_path, 'rb') as f:
        conf = yaml.load(f.read(), Loader=yaml.Loader)

    conf['model_name'] = model
    # Dynamically find the model checkpoint path
    model_path = Path(conf['model_path']) / model
    folders = [f for f in model_path.iterdir() if f.is_dir()]
    if len(folders) > 1:
        raise ValueError(f"Multiple folders found in {model_path}. Please ensure there is only one folder containing the model checkpoints.")
    model_path = model_path / folders[0]
    ckpt = list(model_path.glob('*.pt'))[0]
    model_path = model_path / ckpt.name
    conf['model_path'] = str(model_path)

    with open(config_path, 'w') as f:
        yaml.dump(conf, f)

    result = command_runner.run(command)