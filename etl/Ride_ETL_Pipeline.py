from src.utils import verify_schema
from src.extract import GpxExtractor
import yaml
import os
from pathlib import Path
import subprocess
import argparse
from typing import Optional

FILE_DIR = Path(__file__).parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser('This pipeline runs the ETL process to extract ride data from .gpx files and enrich it for use.')
    parser.add_argument('-c', '--config', default=FILE_DIR / 'Ride_ETL_config.yaml', type=str, help='Enter the filepath of the pipeline config.yaml')
    parser.add_argument('-s', '--secrets', default=FILE_DIR / 'sensitive_config/secrets.yaml', 
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
        config_yaml = verify_schema('config', config_yaml)
    with open(secrets_path, 'r') as f:
        secrets_yaml = yaml.safe_load(f)
        config_yaml = verify_schema('secrets', secrets_yaml)

    # #####################################################################################################
    # # 1. Run the Extraction
    # #####################################################################################################
    # config_extract = config_yaml['extraction']
    # if config_extract['enable'] == True:
    #     if config_extract['clear_outputs']:
    #         output_dir = Path(config_extract['output_directory'])
    #         subprocess.run(f'rm -rf {output_dir}/*', shell=True)

    #     extractor = GpxExtractor(config=config_extract)
    #     extractor.run()
    

    # #####################################################################################################
    # # 2. Run the Transformation
    # #####################################################################################################
    # config_transform = config_yaml['transformation']
    # if config_transform['enable'] == True:
    #     if config_extract['clear_outputs']:
    #         output_dir = Path(config_extract['output_directory'])
    #         subprocess.run(f'rm -rf {output_dir}/*', shell=True)

