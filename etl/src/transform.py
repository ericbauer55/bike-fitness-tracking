import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import reduce
from haversine import haversine
from typing import Optional
from pathlib import Path
from .utils import read_ride_csv
from .clean import PrivacyZoner, NoiseFilter

class GpxTransformer:
    def __init__(self, config:dict, privacy_config:dict):
        self.config = config   
        self.privacy_config = privacy_config
        self.df_summary = pd.read_csv(config['summary_input_directory'] / "ride_summary.csv")
        self.Timer: TimeUpsampler = None
        self.Enricher: BasicEnricher = None
        self.NoiseFilter: NoiseFilter = None
        self.PowerEstimator: PowerEstimator = None
        self.PrivacyScrubber: PrivacyZoner = None
        self.transformers: dict = self._initialize_transformers()

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
        for file in tqdm(file_list):
            ride_id = file.stem # 'path/to/my/ride_001.csv --> has stem=='ride_001'
            df_ride = self._read_ride_csv(file)
            
            # Transform each Ride File
            df_ride = self._process_transforms(df_ride, ride_id)
            df_ride.to_csv(self.config['output_directory'] / f'{ride_id}.csv', index=False)

            # After all of the transformations, summarize the ride
            ride_summary_datum = self._get_ride_summary(df_ride, ride_id)
            summary_data.append(ride_summary_datum)
        
        df_summary_new = pd.DataFrame(summary_data)
        self._save_new_ride_summary(df_summary_new)

    #######################################################################################
    # Helper Methods
    #######################################################################################
    def _initialize_transformers(self) -> dict:
        self.Timer = TimeUpsampler()
        self.Enricher = BasicEnricher()
        self.NoiseFilter = NoiseFilter()
        self.PowerEstimator = PowerEstimator(self.config['power_params'])
        self.PrivacyScrubber = PrivacyZoner(self.privacy_config)

    def _process_transforms(self, df:pd.DataFrame, ride_id:str) -> pd.DataFrame:
        df = self.Timer.process(df, time_gap_threshold=self.config['time_gap_threshold'], upsample=True)
        df = self.Enricher.process(df)
        columns_to_filter = ['ambient_temp_C','heart_rate_bpm','cadence_rpm','speed', 'grade', 'grade_saturated']
        df = self.NoiseFilter.filter_columns(df, columns_to_filter)
        # Get the total weight for this ride
        total_weight = self.df_summary.loc[self.df_summary['ride_id']==ride_id, [col for col in self.df_summary.columns if 'weight' in col]].values.sum()
        df = self.PowerEstimator.estimate_power(df, total_weight)
        df = self.PrivacyScrubber.process(df)
        return df

    def _read_ride_csv(self, file_path:str, time_columns:list[str]=None) -> pd.DataFrame:
        if time_columns is None: time_columns=['time']
        # Read in the CSV file for the Ride
        df = pd.read_csv(file_path)
        
        # guarantee the timestamps are datetime objects
        for time_col in time_columns:
            df[time_col] = pd.to_datetime(df[time_col])

        # drop columns which typically don't have useful information
        drop_columns = ['track_id']
        df = df.drop(columns=drop_columns)

        # ensure all dataframes have sensor columns -- even if they are null
        add_null_columns = ['atemp', 'hr', 'cad'] # if these columns don't already exist add them as np.null

        for col in add_null_columns:
            if col not in df.columns:
                df[col] = np.nan

        # ensure for all numerical columns that their type is truly a float
        for col in df.columns:
            if col in ['time','segement_id','ride_id']: continue # skip this
            df[col] = df[col].astype(float)

        # rename columns for easier reading later
        df = df.rename(columns={'atemp':'ambient_temp_C','hr':'heart_rate_bpm', 'cad':'cadence_rpm'})

        return df
    
    
    def _get_ride_summary(self, df:pd.DataFrame, ride_id:str) -> dict:
        # TODO: get the average speed, average cruising speed, total time vs moving time, total distance
        #               heart rate zones, power zones, avg cadence, cadence_duty_cycle
        feet_to_miles = 1.0 / 5280.0
        C_to_F = lambda c: (9.0/5.0)*c + 32.0
        summary = {'ride_id':ride_id,
                   'avg_speed':df['filt_speed'].mean(),
                   'avg_cruising_speed':df.loc[df['is_cruising']==True, 'filt_speed'].mean(),
                   'total_ride_time_sec':df['elapsed_time'].max(),
                   'total_moving_time_sec':df.loc[df['is_cruising']==True,'delta_time'].sum(),
                   'total_distance_mi':feet_to_miles * df['delta_dist_ft'].sum(), 
                   'total_ascent_ft':df['elapsed_ascent'].max(),
                   'total_descent_ft':df['elapsed_descent'].max(),
                   'avg_heart_rate':df['heart_rate_bpm'].mean(),
                   'avg_power':df['inst_power'].mean(),
                   'avg_cadence':df['filt_cadence_rpm'].mean(),
                   'avg_ambient_temp_F':df['ambient_temp_C'].apply(C_to_F).mean()
                   #'cadence_duty_cycle':sum((df['is_cruising']==True)&(df['filt_cadence']>20)) / sum(df['is_cruising']==True)
                   }
        power_summary = self._calculate_power_curve(df)
        summary |= power_summary
        
        return summary
    
    def _calculate_power_curve(self, df: pd.DataFrame):
        # Create a set of rolling windows to calculate a MAX over avg(inst_powers[within_window])
        rolling_windows = [4, 5, 10, 20, 30, 60, # seconds
                            2*60, 3*60, 4*60, 5*60, 6*60, 10*60, 20*60, 30*60, 40*60, # minutes
                            60*60, 2*60*60, 3*60*60, 4*60*60] # hours
        rwindow_labels = ['4s', '5s', '10s', '20s', '30s', '1m', # seconds
                            '2m', '3m', '4m', '5m', '6m', '10m', '20m', '30m', '40m', # minutes
                            '1h', '2h', '3h', '4h'] # hours
        label_map = {seconds:label for seconds,label in zip(rolling_windows,rwindow_labels)}

        # Initialize a list to store the peak powers per window
        window_peak_powers = []

        for rwindow in rolling_windows:
            # We should not calculate the peak power for a window that is longer than this value since it is ill defined
            rolling_avg_inst_power = df[['inst_power']].rolling(rwindow, min_periods=rwindow).mean().dropna()
            if rolling_avg_inst_power.shape[0]==0: # all values were np.nan, hence the window is too large for the ride data
                peak_power = np.nan
            else:
                peak_power = max(rolling_avg_inst_power.values)[0]
            window_peak_powers.append({'time_window':label_map[rwindow], 'window_length_seconds':rwindow, 'peak_avg_power':peak_power})
        
        # Create and output dictionary that summarizes the best power efforts by duration for this ride
        df_pwr = pd.DataFrame(window_peak_powers)
        output_summary = dict()
        for _, row in df_pwr[['time_window','peak_avg_power']].iterrows():
            output_summary[f'best_power_{row["time_window"]}'] = row['peak_avg_power']
        return output_summary

    def _save_new_ride_summary(self, df_summary_new:pd.DataFrame) -> None:
        # Merge new summary data into old summary data using ride_id as the key
        df_total_summary = self.df_summary.merge(df_summary_new, how='left',on='ride_id')
        df_total_summary.to_csv(self.config['summary_output_directory'] / f'ride_summary.csv', index=False)



################################################################################################################

class TimeUpsampler:
    def process(self, df:pd.DataFrame, time_gap_threshold:int=15, upsample:bool=False) -> pd.DataFrame:
        df = df.copy()
        df['is_original_row'] = True
        df = self.enrich_time_data(df)
        df['elapsed_time'] = df['delta_time'].cumsum()
        df = self.label_continuous_segments(df, time_gap_threshold)
        if upsample:
            return self.upsample(df)
        else:
            return df

    def upsample(self, df:pd.DataFrame) -> pd.DataFrame:
        df_upsampled = self.normalize_sampling_rate(df)
        df_upsampled['ride_id'] = df.loc[0,'ride_id'] # copy the ride_id into all interpolated rows
        df['is_original_row'] = df['is_original_row'].fillna(False) # these are interpolated rows
        return df_upsampled

    @staticmethod
    def enrich_time_data(df:pd.DataFrame, time_column:str='time', fill_first:float=1.0):
        df = df.copy()
        # Temporarily get the number of seconds since Jan. 1, 1970 as the UTC timestamp
        df['time_utc'] = df[time_column].apply(lambda x: x.timestamp())
        
        # Calculate the row-wise difference in time (in seconds)
        df['delta_time'] = df['time_utc'].diff()
        
        # drop the temporary time column
        df.drop(['time_utc'], axis=1, inplace=True)
        
        # fill in the initial value of delta_time with @fill_first
        df['delta_time'] = df['delta_time'].fillna(fill_first)
        
        return df
    
    @staticmethod
    def label_continuous_segments(df: pd.DataFrame, time_gap_threshold:int=15):
        df = df.copy()
        # get the time gap indices
        # Calculate when the time discontinuities occur
        filt_time_jump = df['delta_time'] >= time_gap_threshold
        time_gap_indices = list(df.loc[filt_time_jump, 'time'].index)

        # intialize the initial segment_id. to be incremented for each region of continuous data
        segment_id_counter = 0
        # initialize the starting index of the first segment
        segment_start_index = 0

        for time_gap_index in time_gap_indices:
            # Assign the Segment ID
            df.loc[segment_start_index:time_gap_index-1, 'segment_id'] = segment_id_counter
            
            # update the segment_id counter and start index
            segment_id_counter += 1
            segment_start_index = time_gap_index
            
        # Since segment_id == -1 by default, this represents the final segment of activity once parsed
        df['segment_id'] = df['segment_id'].replace({-1:segment_id_counter})

        return df
    
    @staticmethod
    def upsample_and_interpolate(df: pd.DataFrame, time_column:str='time', method:str='linear', limit_direction:str='forward'):
        # Since the delta_time column is no longer needed to detect discontinuities,
        # Drop delta_time so we can rebuild it at a segment_id level
        df.drop(['delta_time'], axis=1, inplace=True)
        
        # set the timestamp as the index for the dataframe
        kwargs = dict(method=method, limit_direction=limit_direction)
        if method=='spline':
            kwargs['order']=2
        # Get rid of any duplicate rows to prevent reindexing errors
        df = df.drop_duplicates(subset=[time_column])

        # Resample the rows to a 1 Hz sampling rate (1 second sampling period)
        df = df.set_index(time_column).copy()
        df = df.resample('s').interpolate(**kwargs).reset_index()
        return df
    
    def normalize_sampling_rate(self, df: pd.DataFrame, partition_column:str='segment_id') -> pd.DataFrame:
        df = df.copy()
        functions_to_apply = [self.upsample_and_interpolate, self.enrich_time_data]
        for func in functions_to_apply:
            df = pd.concat(list(map(func, [df_group for _,df_group in df.groupby(partition_column)])), ignore_index=True).sort_index()
        return df
    

class BasicEnricher:
    def process(self, df:pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        process_steps = [self.compute_distance, self.compute_heading, self.compute_speed, self.flag_cruising_rows, 
                         self.convert_elevation, self.compute_grade, self.compute_cumulative_elevation_changes]
        
        df_enriched = reduce(lambda df,func:func(df), process_steps, df)
        return df_enriched

    ##############################################################################################
    # Distance and Heading from Lat/Long
    ##############################################################################################
    @staticmethod
    def compute_distance(df, latitude='latitude', longitude='longitude', fill_first=0.0):
        df = df.copy()
        # Copy the previous values of Lat/Long to the current row for vectorized computation
        df['lat_old'] = df[latitude].shift()
        df['long_old'] = df[longitude].shift()
        
        # Grab the relevant columns for distance calculation
        df_gps = df[['lat_old', 'long_old', latitude, longitude]]
        
        # Define an anonymous function to execute over each row to calculate the distance between rows
        # Units should be in feet to prevent floating point precision issues for other calculations. (lower noise)
        haversine_distance = lambda x: max(0,haversine((x.iloc[0], x.iloc[1]), (x.iloc[2], x.iloc[3]), unit='ft'))
        
        # Create the distance column, making sure to apply the function row-by-row
        df['delta_dist_ft'] = df_gps.apply(haversine_distance, axis=1)
        df['delta_dist_ft'] = df['delta_dist_ft'].fillna(fill_first)
        
        # Remove the old latitude and longitude columns
        df.drop(['lat_old','long_old'], axis=1, inplace=True)
        return df
    
    @staticmethod
    def compute_heading(df, latitude='latitude', longitude='longitude'):
        df = df.copy()
        # Copy the previous values of Lat/Long to the current row for vectorized computation
        df['lat_old'] = df[latitude].shift()
        df['long_old'] = df[longitude].shift()
        
        # Grab the relevant columns for distance calculation
        df_gps = df[['lat_old', 'long_old', latitude, longitude]]
        
        # Define an anonymous function to execute over each row to calculate the angle with North as 0 degrees
        # NOTE: we use "delta_lat / delta_long" to ensure that North = 0 degrees
        rad2deg = 180.0 / np.pi
        heading = lambda x: rad2deg * np.arctan2((x.iloc[2]-x.iloc[0]), (x.iloc[3]-x.iloc[1])) # atan(delta_lat / delta_long)
        
        # Create the distance column, making sure to apply the function row-by-row
        df['heading'] = df_gps.apply(heading, axis=1)
        df['heading'] = df['heading'].apply(lambda x: x + 360.0*(1-np.sign(x))/2) # correct for negative angles
        
        # Remove the old latitude and longitude columns
        df.drop(['lat_old','long_old'], axis=1, inplace=True)
        return df

    ##############################################################################################
    # Speed Enrichments
    ##############################################################################################
    @staticmethod
    def compute_speed(df):
        df = df.copy()
        feet_to_miles = 1.0 / 5280.0
        miles_per_second_2_MPH = 3600.0 / 1.0 # conversion factor
        df['speed'] = miles_per_second_2_MPH * (feet_to_miles*df['delta_dist_ft']) / df['delta_time']
        return df

    @staticmethod
    def flag_cruising_rows(df, start_threshold_mph:float=8.0, stop_threshold_mph:float=5.0):
        """
        Scmitt Trigger to implement a hysteresis state machine for determining a state
        """
        df = df.copy()
        df['is_cruising'] = False

        for k in range(1, df.shape[0]):
            previous_state = df.loc[k-1,'is_cruising']
            current_speed = df.loc[k,'speed']
            if (previous_state==False) & (current_speed >= start_threshold_mph):
                df.loc[k,'is_cruising'] = True # rising threshold surpassed
            elif (previous_state==True) & (current_speed < stop_threshold_mph):
                df.loc[k,'is_cruising'] = False # rising threshold surpassed
            else:
                # if there is no change, propogate the previous state
                df.loc[k,'is_cruising'] = df.loc[k-1,'is_cruising']
        return df

    ##############################################################################################
    # Elevation Enrichments
    ##############################################################################################
    @staticmethod
    def convert_elevation(df):
        df = df.copy()
        meters_to_feet = 3.281
        df['elevation'] = df['elevation'] * meters_to_feet
        return df

    @staticmethod
    def compute_grade(df):
        df = df.copy()
        fill_first = 0.0
        df['delta_ele_ft'] = df['elevation'].diff()
        df['delta_ele_ft'] = df['delta_ele_ft'].fillna(fill_first)
        df['grade'] = 100.0 * (df['delta_ele_ft'] / df['delta_dist_ft'])
        df.loc[0,'grade'] = 0.0 # initialize and assumed 0% slope as the starting point--representative of a typical parking lot
        df.loc[~np.isfinite(df['grade']),'grade'] = np.nan

        # fill in nulls where delta_dist==0.0 by interpolating the value. If you stop on a hill, your grade should carry forward
        df['grade'] = df['grade'].interpolate('linear')

        # Constrain Grade to be within the typical +/- 15 % all riders deal with. We'll use 18% as thresholds
        df['grade_saturated'] = df['grade'].apply(lambda g: min(g,18)).apply(lambda g: max(g,-18))

        return df

    @staticmethod
    def compute_cumulative_elevation_changes(df, fill_first=0.0):
        df = df.copy()
        
        # create an elevation difference
        df['delta_ele'] = df['elevation'].diff()
        df['delta_ele'] = df['delta_ele'].fillna(fill_first)
        
        # create delta ascent and delta descent columns
        df['delta_ascent'] = df.loc[df['delta_ele']>=0, 'delta_ele']
        df['delta_descent'] = df.loc[df['delta_ele']<=0, 'delta_ele']
        
        # create the cumulative versions
        df['elapsed_ascent'] = df['delta_ascent'].cumsum()
        df['elapsed_ascent'] = df['elapsed_ascent'].interpolate() # fill in any blanks
        df['elapsed_descent'] = df['delta_descent'].cumsum()
        df['elapsed_descent'] = np.abs(df['elapsed_descent'].interpolate()) # fill in any blanks
        
        # create the total elevation change column
        df['elapsed_elevation'] = df['elapsed_ascent'] + df['elapsed_descent']
            
        # drop the elevation differences
        df.drop(['delta_ele','delta_ascent','delta_descent'], axis=1, inplace=True)
        
        return df

class PowerEstimator:
    def __init__(self, power_params:dict, speed_col:str='filt_speed', grade_col:str='filt_grade_saturated')->None:
        self.power_params = power_params
        self.speed_col = speed_col
        self.grade_col = grade_col

    def estimate_power(self, df:pd.DataFrame, total_weight_lbs:float) -> pd.DataFrame:
        """
        @total_weight_lbs should be calculated from the ride summary weight categories (rider, bike, bags, etc)
        """
        pounds_to_kilograms = 0.453592
        total_mass_kg = pounds_to_kilograms * total_weight_lbs
        df = df.copy()
        return self.get_instantaneous_power(df, self.power_params, total_mass_kg, self.speed_col, self.grade_col)

    @staticmethod
    def get_instantaneous_power(df:pd.DataFrame, power_params:dict, total_mass:float,
                                speed_column:str='speed', grade_column:str='grade_saturated'):
        cols_to_drop_later = ['grade_radians','speed_MpS','F_grav','F_fric','F_drag', 'F_sum', 
                            'total_speed']
        params = power_params

        # Convert the terrain slope into radians
        df['grade_radians'] = np.arctan(df[grade_column]/100)
        
        # Convert the speed units into meters per second
        mph2MpS = 0.44704 # 1 MPH = 0.44704 m/s
        df['speed_MpS'] = mph2MpS * df[speed_column]
        
        # Get the total speed component with wind (placeholder)
        df['total_speed'] = df['speed_MpS']
        
        # Calculate the individual forces
        df['F_grav'] = total_mass*params['gravity'] * np.sin(df['grade_radians'])
        df['F_fric'] = params['mu_rr']*total_mass*params['gravity'] * np.cos(df['grade_radians'])
        full_coefficient = 0.5 * params['rho_air'] * params['area'] * params['c_drag']
        df['F_drag'] = (full_coefficient) * np.power(df['total_speed'], 2) # k(v)^2
        
        # Sum the forces
        df['F_sum'] = df['F_drag'] + df['F_grav'] + df['F_fric']
        
        # Calculate the non-negative power delivered by the ride (set Power=0 for F_sum <0)
        df['inst_power'] = (1.0/params['eta_dt']) * df['F_sum'] * df['speed_MpS'] 
        df.loc[df['inst_power']<0,'inst_power'] = 0 # coasting when sum of forces is negative (no input power)

        df.drop(columns=cols_to_drop_later, inplace=True)
        
        return df