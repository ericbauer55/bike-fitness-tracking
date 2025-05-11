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

        summary_data = []

        # For each file, run the extraction process
        for file in file_list:
            ride_id = 'temporary'
            df_ride = read_gpx_to_dataframe(file, ride_id)
            ride_id = self._get_ride_id(df_ride.loc[0,'time'])
            df_ride['ride_id'] = ride_id
            df_ride.to_csv(self.config['output_directory'] / f'{ride_id}.csv', index=False)

            # Record a brief summary of each ride to start. It will be enhanced later
            df_ride['time'] = pd.to_datetime(df_ride['time'])
            summary_datum = {'ride_id':ride_id, 'start_date':df_ride.loc[0,'time'].date(), 
                             'start_time':df_ride.loc[0,'time'].time(), 'end_time':df_ride.loc[df_ride.shape[0]-1,'time'].time(),
                             'biker_weight_lbs':220, 'bike_weight':25, 'bag_weight':5}
            summary_data.append(summary_datum)
        df_summary = pd.DataFrame(summary_data).sort_values(by=['start_date','start_time'])
        df_summary.to_csv(self.config['summary_output_directory'] / f'ride_summary.csv', index=False)

    #######################################################################################
    # Helper Methods
    #######################################################################################
    @staticmethod
    def _get_ride_id(start_time:pd.Timestamp) -> str:
        return str(hex(hash(start_time.timestamp())))