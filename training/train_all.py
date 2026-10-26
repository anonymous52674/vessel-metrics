"""Run training for all models defined in this file with configuration defined in configs/train.yml"""

import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

import yaml

@dataclass
class CommandResult:
    command: List[str]
    returncode: int
    output: str
    
    @property
    def success(self) -> bool:
        return self.returncode == 0

class CommandRunner:
    def __init__(self, stream_realtime: bool = True, log_filepath: Optional[str] = None):
        """
        Args:
            stream_realtime: If True, prints output to the console as it happens.
            log_filepath: If provided, appends all output to this file.
        """
        self.stream_realtime = stream_realtime
        self.log_filepath = log_filepath

        if self.log_filepath:
            with open(self.log_filepath, 'w', encoding='utf-8') as f:
                f.write("--- Command Runner Log Started ---\n")

    def run(self, command: List[str]) -> CommandResult:
        output_lines = []
        
        log_file = open(self.log_filepath, 'a', encoding='utf-8') if self.log_filepath else None

        try:
            if log_file:
                log_file.write(f"\n--- Running: {' '.join(command)} ---\n")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merges errors into standard output stream
                text=True,
                bufsize=1 # Line-buffered so we get data immediately
            )

            for line in process.stdout:
                output_lines.append(line)
                
                # Print to console
                if self.stream_realtime:
                    sys.stdout.write(line)
                    sys.stdout.flush() 
                    
                # Write to log file
                if log_file:
                    log_file.write(line)
                    log_file.flush()

            process.wait()
            full_output = "".join(output_lines)
            
            return CommandResult(
                command=command,
                returncode=process.returncode,
                output=full_output
            )

        except FileNotFoundError:
            error_msg = f"ERROR: Command not found: {command[0]}\n"
            if self.stream_realtime: sys.stdout.write(error_msg)
            if log_file: log_file.write(error_msg)
            return CommandResult(command, 127, error_msg)

        finally:
            if log_file:
                log_file.close()

                    # Memory usage on DRIVE dataset with batch size 2 (in GB):
models = [
    'LWUNet',
    'SAUNet',       # 0.31 GB
    'UNet',         # 1.08 GB
    'ResUNet2P',    # 1.65 GB
    'ConvUNeXt',    # 1.65 GB
    'DCSAU_UNet',   # 1.67 GB
    'ResUNet',      # 1.76 GB
    'ATTUNet',      # 1.83 GB
    'AGNet',        # 1.92 GB
    'FRUNet',       # 2.14 GB
    'UNet2P',       # 2.70 GB
    'R2UNet',       # 3.82 GB
    'FSGNet_noGRM', # 4.06 GB
    'UNet3P_Deep',  # 4.57 GB   
    'HRNet_t',      # 5.46 GB
    'FSGNet',       # 6.53 GB
]

if __name__ == "__main__":

    base_config_path = "configs/train.yml"
    config_path = "configs/train_model.yaml"
    command = ["python", "-u", "main.py", "--config_path", config_path]

    command_runner = CommandRunner(stream_realtime=True, log_filepath="train_all.log")
    for model in models:

        with open(base_config_path, 'rb') as f:
            conf = yaml.load(f.read(), Loader=yaml.Loader)
            conf['model_name'] = model

        with open(config_path, 'w') as f:
            yaml.dump(conf, f)

        result = command_runner.run(command)
    