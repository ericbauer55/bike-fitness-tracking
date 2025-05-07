from ride_extractor.ride_extractor.src.extract import GpxExtractor
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
        config = yaml.safe_load(f)
    with open(secrets_path, 'r') as f:
        secrets = yaml.safe_load(f)

    #####################################################################################################
    # 1. Run the Extraction
    #####################################################################################################
    if extract_enable:=config.get('extraction',dict()).get('enable',None) == True:
        pass
    elif extract_enable is None:
        raise KeyError('Extraction config lacking proper schema.')
    

    #####################################################################################################
    # 2. Run the Transformation
    #####################################################################################################
    if transform_enable:=config.get('transformation',dict()).get('enable',None) == True:
        pass
    elif transform_enable is None:
        raise KeyError('Transformation config lacking proper schema.')

