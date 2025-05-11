import pandas as pd
import gpxpy as gx
from pathlib import Path
from schema import Schema, And, Or, Use

def read_gpx_to_dataframe(file_path:str, ride_id:str)->pd.DataFrame:
    """
    Given a fully qualified @file_path, this function utilizes the gpxpy
    package to parse the Track Point data from the XML structure.

    This returns a Pandas dataframe of the GPX file. If any sensor values are present,
    it will read those in as new columns from the extensions columns.
    As such, the base schema = ['ride_id', 'track_id', 'segment_id', 'time', 'elevation', 'latitude', 'longitude']

    Note that due to the way Strava captures activities, track_id and segment_id columns will just be id=0 for all rows
    """
    if not Path(file_path).exists(): raise FileNotFoundError(f'The file "{file_path}" does not exist.')

    # Setup data capture as lists initially
    data = []

    # Open up the .gpx file and gather each point of data
    with open(file_path,'r') as opened_file:
        # parse the .gpx file into a GPX object
        gpx = gx.parse(opened_file)
        
        # iterate through all tracks, segments and points to extract the data
        for i, track in enumerate(gpx.tracks):
            for j, segment in enumerate(track.segments):
                for point in segment.points:
                    # create the row of data & append to data
                    row = {'ride_id':ride_id, 'track_id':i,'segment_id':-1, 'time':point.time, 
                        'elevation':point.elevation, 'latitude':point.latitude, 'longitude':point.longitude}
                    # determine the data available in sensor extension tags (if any)
                    if len(point.extensions)>0:
                        row_extension = dict()
                        for element in point.extensions[0]:
                            tag = element.tag.split('}')[-1] # remove the {schema_prefix_url} that prepends the extension name
                            row_extension[tag] = element.text
                        
                        row |= row_extension
                    data.append(row)
    
    # Capture the data structure as a Pandas Dataframe
    df = pd.DataFrame(data)

    return df


def read_ride_csv(file_path:str, time_columns=['time'])->pd.DataFrame:
    """
    This function loads in a ride's data from .CSV given a file path as a dataframe

    The state of the data (processed vs. enriched) doesn't matter as long
    as there is a 'time' column for the timestamp
    """
    if not Path(file_path).exists(): raise FileNotFoundError(f'The file "{file_path}" does not exist.')

    # Read in the CSV file for the Ride
    df = pd.read_csv(file_path)
    
    # guarantee the timestamps are datetime objects
    for time_col in time_columns:
        df[time_col] = pd.to_datetime(df[time_col])

    return df

def verify_schema(config_type:str, data:dict) -> bool:
    if config_type=='config':
        schema_dict = {And('extraction'):{'enable': And(bool),
                                    'clear_outputs': And(bool),
                                    'input_directory': And(str, Use(lambda x:Path(x).resolve())),
                                    'output_directory': And(str, Use(lambda x:Path(x).resolve())),
                                    'summary_output_directory': And(str, Use(lambda x:Path(x).resolve()))
                                    },
                        And('transformation'):{'enable': And(bool),
                                        'clear_outputs': And(bool),
                                        'input_directory': And(str, Use(lambda x:Path(x).resolve())),
                                        'output_directory': And(str, Use(lambda x:Path(x).resolve())),
                                        'summary_input_directory': And(str, Use(lambda x:Path(x).resolve())),
                                        'summary_output_directory': And(str, Use(lambda x:Path(x).resolve())),
                                        'time_gap_threshold':And(float, lambda x:x>0),
                                        'scrub_private_coordinates': And(bool),
                                        And('power_params'):{'area': And(float, Use(float), lambda x: x>0),
                                                             'mu_rr': And(float, Use(float), lambda x: x>0 and x<0.5),  
                                                             'c_drag': And(float, Use(float), lambda x: x>0), 
                                                             'rho_air': And(float, Use(float), lambda x: x>0),   
                                                             'eta_dt': And(float, Use(float), lambda x: x>0 and x<1.0),  
                                                             'gravity': And(float, Use(float), lambda x: x in [9.8, 32.15224]) # m/s^2, ft/s^2
                                                            }
                                        }
                        }

    elif config_type=='secrets':
        schema_dict = {'home_location':{'lat':And(float, lambda x: -90 <= x <= 90),
                                        'long':And(float, lambda x: -180 <= x <= 180),
                                        'scrub_radius':And(Or(float,int), lambda x:x>0)}}
    else:
        raise ValueError(f'Only types ["config","secrets"] are allowed. Type "{config_type}" is invalid.')

    schema = Schema(schema_dict)
    return schema.validate(data)