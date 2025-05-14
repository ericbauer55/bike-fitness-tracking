from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import great_expectations as gx
import great_expectations.expectations as gxe
from functools import reduce
import json
# ref: https://docs.greatexpectations.io/docs/core/introduction/


class CleanRideLoader:
    def __init__(self, config:dict):
        self.config = config
        # top 1% of cyclists, ref: https://www.cyclinganalytics.com/blog/2018/06/how-does-your-cycling-power-output-compare
        self.top_cyclist_powers = {'best_power_4s':1650, 'best_power_5s':1500,'best_power_10s':1375, 'best_power_20s':1200, 
                              'best_power_30s':1000, 'best_power_1m':775, 'best_power_2m':600, 'best_power_3m':560, 
                              'best_power_4m':535, 'best_power_5m':510, 'best_power_6m':500, 'best_power_10m':490, 
                              'best_power_20m':460, 'best_power_30m':420, 'best_power_40m':375, 'best_power_1h':360}
        self.df_summary:pd.DataFrame = self._load_summary(config['summary_input_directory'])

    def run(self)->None:
        # 1. Determine which ride_id's are invalid due to data quality violations
        violation_indices, violation_details = self._get_data_quality_violations()

        # 2. Split the summary table into a violations partition and a non-violations partition
        df_summary_good = self.df_summary.copy().loc[~self.df_summary.index.isin(violation_indices),:].reset_index(drop=True)
        df_summary_bad = self.df_summary.copy().loc[self.df_summary.index.isin(violation_indices),:].reset_index(drop=True)
        with open(Path(self.config['summary_output_directory']) / 'violations.json', 'w') as f:
            json.dump(violation_details, f, indent=4)
        

        # 3. Save the summary tables
        df_summary_good.to_csv(Path(self.config['summary_output_directory']) / 'ride_summary_good.csv', index=False)
        df_summary_bad.to_csv(Path(self.config['summary_output_directory']) / 'ride_summary_violations.csv', index=False)

        # 4. Load the transformed ride data for only the ride_id's without violations
        for ride_id in tqdm(df_summary_good['ride_id']):
            df_ride = pd.read_csv(Path(self.config['input_directory']) / f'{ride_id}.csv')
            df_ride.to_csv(Path(self.config['output_directory']) / f'{ride_id}.csv', index=False)
    

    #######################################################################################
    # Helper Methods
    #######################################################################################
    def _load_summary(self, summary_path:str)->pd.DataFrame:
        df_summary = pd.read_csv(Path(summary_path) / 'ride_summary.csv')
        df_summary['start_date'] = pd.to_datetime(df_summary['start_date'])
        df_summary['start_time'] = pd.to_timedelta(df_summary['start_time'])
        df_summary['end_time'] = pd.to_timedelta(df_summary['end_time'])
        return df_summary
    
    def _get_data_quality_violations(self) -> tuple[list[int], dict[str,list[int]]]:
        ## 1. Setup the Great Expectations context --> batch objects
        context = gx.get_context(mode='ephemeral') 
        data_source_name = 'ride_summary_data'
        data_source = context.data_sources.add_pandas(name=data_source_name)
        data_asset_name = 'summary_dataframe'
        data_asset = data_source.add_dataframe_asset(name=data_asset_name)
        batch_definition_name = 'whole_summary_dataframe'
        batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)
        batch_parameters = {"dataframe": self.df_summary}
        # Get the whole dataframe as a Batch
        batch = batch_definition.get_batch(batch_parameters=batch_parameters)

        ## 2a. Create and Validate Expectations for Power Columns 
        # violation indices
        violations = dict()
        expectations = dict()
        # Create an Expectation to test
        for col in self.top_cyclist_powers.keys():
            expectations[col] = gxe.ExpectColumnValuesToBeBetween(column=col, max_value=self.top_cyclist_powers[col], min_value=0)
            validation_results = batch.validate(expectations[col], **{"result_format": "COMPLETE"})
            violations[col] = validation_results['result']['unexpected_index_list']

        ## 2b. Create and Validate the minimum ride time duration expectation
        duration_col = 'total_ride_time_sec'
        # The minimum ride should be at least 5 minutes and less than 24 hours
        # TODO: Fix this expectation... its giving too many false positives
        # expectations[duration_col] = gxe.ExpectColumnValuesToBeBetween(column=col, max_value=60*60*24, 
        #                                                                min_value=self.config['min_ride_time_duration_seconds'])
        # validation_results = batch.validate(expectations[duration_col] , **{"result_format": "COMPLETE"})
        # violations[duration_col] = validation_results['result']['unexpected_index_list']

        ## 3a. Get the UNION'd set of violation indices
        violations_indices = list(set(reduce(lambda x,y: x+y,violations.values(), [])))
        violations_indices.sort()

        ## 3b. Get the Ride IDs for each set of violations
        violation_details = {col:[] for col in violations.keys()}
        for col, indices in violations.items():
            for index in indices:
                bad_ride_id = self.df_summary.loc[index,'ride_id']
                violation_details[col].append(bad_ride_id)

        return violations_indices, violation_details