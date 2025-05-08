from .utils import read_gpx_to_dataframe
from typing import Optional, Any
from pathlib import Path

class GpxExtractor:
    def __init__(self, config:dict):
        self.config = config        

    def run(self, file_list: Optional[list[str]]=None) -> None:
        if file_list is None:
            file_list = list(self.config['input_directory'].iterdir())
        else:
            file_list = list(map(Path, file_list))
            # Verify that all files exist
            f_exists = [file.exists() for file in file_list]
            if not all(f_exists):
                bad_files = list(filter(lambda x:x==True, f_exists))
                raise FileNotFoundError(f'The following files do not exist: {"\n\t".join(bad_files)}')

        # For each file, run the extraction process
        for file in file_list:
            ride_id = file.stem # 'path/to/my/ride_001.gpx --> has stem=='ride_001'
            df_ride = read_gpx_to_dataframe(file, ride_id)
            df_ride.to_csv(self.config['output_directory'] / f'{ride_id}.csv')

    #######################################################################################
    # Helper Methods
    #######################################################################################