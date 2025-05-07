from ride_extractor.ride_extractor.src.extract import GpxExtractor
from ride_extractor.ride_extractor.src.utils import verify_schema
import yaml
import os
from pathlib import Path
import subprocess
import argparse
from typing import Optional

PWD = Path().cwd()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('This pipeline runs the ETL process to extract ride data from .gpx files and enrich it for use.')
    parser.add_argument('-c', '--config', default=PWD / 'Ride_ETL_config.yaml', type=str, help='Enter the filepath of the pipeline config.yaml')
    parser.add_argument('-s', '--secrets', default=PWD / 'sensitive_config/secrets.yaml', 
                        type=Optional[str], help='Enter the path for sensitive config params')
    
    args = parser.parse_args()
    config_path = args.config
    secrets_path = args.secrets
    
    # Confirm the files exist
    if not Path(config_path).exists(): raise FileNotFoundError(f'The config file "{config_path}" does not exist.')
    if not Path(secrets_path).exists(): raise FileNotFoundError(f'The secrets file "{secrets_path}" does not exist.')
    
    # Load the files
    with open(config_path, 'r') as f:
        config_yaml = yaml.safe_load(f)
        verify_schema('config', config_yaml)
    with open(secrets_path, 'r') as f:
        secrets_yaml = yaml.safe_load(f)
        verify_schema('secrets', secrets_yaml)

    #####################################################################################################
    # 1. Run the Extraction
    #####################################################################################################
    config_extract = config_yaml['extraction']
    if config_extract['enable'] == True:
        if config_extract['clear_outputs']:
            output_dir = Path(config_extract['output_directory'])
            subprocess.run(f'', shell=True) # TODO: finish this directory removal

        extractor = GpxExtractor(config=config_extract)
        extractor.run()
    

    #####################################################################################################
    # 2. Run the Transformation
    #####################################################################################################
    config_transform = config_yaml['transformation']
    if config_transform['enable'] == True:
        if config_extract['clear_outputs']:
            output_dir = Path(config_extract['output_directory'])
            subprocess.run(f'', shell=True) # TODO: finish this directory removal

