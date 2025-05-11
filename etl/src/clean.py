from os import stat
import pandas as pd
import numpy as np
from haversine import haversine
from scipy import signal
import numpy as np

class NoiseFilter:
    def __init__(self, filter_type:str='hann', filter_order:int=21):
        if filter_type=='hann':
            fir_filter = signal.windows.hann(filter_order)
        elif filter_type=='blackman-harris':
            fir_filter = signal.windows.blackmanharris(filter_order)
        elif filter_type=='rect':
            fir_filter = signal.windows.boxcar(filter_order)
        self.fir_filter = fir_filter

    def filter_columns(self, df:pd.DataFrame, columns:list[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            df = self.apply_filter(df, col, self.fir_filter)
        return df 

    @staticmethod
    def apply_filter(df:pd.DataFrame, signal_column:str, fir_filter:np.ndarray) -> pd.DataFrame:
        signal_list = list(df[signal_column].values)
        filtered_signal = signal.convolve(signal_list, fir_filter, mode='same') / sum(fir_filter)
        
        df['filt_'+signal_column] = np.nan
        if len(signal_list) == len(filtered_signal):
            df['filt_'+signal_column] = filtered_signal
        else:
            df.loc[1:,'filt_'+signal_column] = filtered_signal
            df.loc[0,'filt_'+signal_column] = df.loc[1,'filt_'+signal_column] # backfill
        return df


class PrivacyZoner():
    def __init__(self, privacy_zone_config:dict):
        self.df = None
        self.privacy_zone_config = privacy_zone_config
        self.df_privacy = None
        self.temporary_prox_columns = []

    def process(self, df:pd.DataFrame) -> pd.DataFrame:
        self.df = df.copy()
        self._read_privacy_zones()
        self._calculate_proximities()
        self._remove_violation_gps_data()
        self._drop_temporary_prox_columns()
        return self.df

    ################################################################
    # PROCESS METHODS
    ################################################################

    def _read_privacy_zones(self):
        data = []
        for private_location in self.privacy_zone_config.keys():
            datum = {'name':private_location, 'latitude':self.privacy_zone_config[private_location]['lat'],
                     'longitude':self.privacy_zone_config[private_location]['long'], 
                     'privacy_radius':self.privacy_zone_config[private_location]['scrub_radius']}
            data.append(datum)
        self.df_privacy = pd.DataFrame(data)

    def _calculate_proximities(self):
        for private_location in range(self.df_privacy.shape[0]):
            # get the relevant parameters
            dist_name = 'prox_' + self.df_privacy.loc[private_location, 'name']
            latitude = self.df_privacy.loc[private_location, 'latitude']
            longitude = self.df_privacy.loc[private_location, 'longitude']
            
            # calculate the proximity
            self.df[dist_name] = self._get_proximity_to_address(latitude, longitude)
            self.temporary_prox_columns.append(dist_name)

    def _remove_violation_gps_data(self):
        for private_location in range(self.df_privacy.shape[0]):
             # get the relevant parameters
            dist_name = 'prox_' + self.df_privacy.loc[private_location, 'name']
            privacy_radius = self.df_privacy.loc[private_location, 'privacy_radius']

            filt_violation = self.df.loc[:,dist_name] <= privacy_radius
            self.df.loc[filt_violation, 'latitude'] = np.nan
            self.df.loc[filt_violation, 'longitude'] = np.nan

    def _drop_temporary_prox_columns(self):
        self.df.drop(self.temporary_prox_columns, axis=1, inplace=True)

    ################################################################
    # HELPER METHODS
    ################################################################

    def _get_proximity_to_address(self, addr_latitude, addr_longitude):
        df_gps = self.df[['latitude', 'longitude']]
        
        # Define an anonymous function to execute over each row to calculate the distance between rows
        haversine_distance = lambda x: haversine((x.iloc[0], x.iloc[1]), (addr_latitude, addr_longitude), unit='mi')
        
        # Create the distance column, making sure to apply the function row-by-row
        proximity = df_gps.apply(haversine_distance, axis=1)
        
        return proximity