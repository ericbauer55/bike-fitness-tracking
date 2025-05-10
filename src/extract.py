from .utils import read_gpx_to_dataframe
from typing import Optional, Any
from pathlib import Path
import pandas as pd

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
            ride_id = 'temporary'
            df_ride = read_gpx_to_dataframe(file, ride_id)
            ride_id = self._get_ride_id(df_ride.loc[0,'time'])
            df_ride['ride_id'] = ride_id
            df_ride.to_csv(self.config['output_directory'] / f'{ride_id}.csv', index=False)

    #######################################################################################
    # Helper Methods
    #######################################################################################
    def _get_ride_id(start_time:pd.Timestamp) -> str:
        return str(hex(hash(start_time.timestamp())))