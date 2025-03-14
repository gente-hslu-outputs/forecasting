"""
Simple Dataset creation from Influx DB
"""
import os
from pathlib import Path
import logging
from datetime import datetime,timezone, timedelta

# Dependencies
import pandas as pd
import numpy as np


# Git Dependencies
from databaseaccessor import DfReader

class Dataset:

    def __init__(self, input_keys:dict,
                 input_window:int,
                 output_key:dict,
                 output_window:int,
                 start_date:datetime,
                 end_date:datetime,
                 norm:str = None,           # Normalisation function ("std", "minmax", None)
                 batch_size:int = 1,
                 config_file_path:str = None) -> None:
        """
        input_keys (dict[str,dict[str,str]]):   Model Inputs (X)
                                                        Format of Dict is: (key_1 = field_name)
                                                            {key_1:
                                                                {"bucket": bucket_name,
                                                                 "measurement": measurement_name},
        """

        self.input_keys = input_keys
        self.input_window = input_window
        self.input_dim = len(self.input_keys)

        self.output_window = output_window
        self.output_key = output_key
        self.output_dim = 1

        # always as UTC
        self.start_date = start_date.replace(tzinfo=None).replace(tzinfo=timezone.utc)
        self.end_date = end_date.replace(tzinfo=None).replace(tzinfo=timezone.utc)

        
        # slicing parameters (how are the datasets sliced)
        self.stride = 1                                 # how many timesteps between slices 
        total_days = (end_date - start_date).days
        self.test_samples = int(total_days / 18)        # 20 test days per year (not overlapping)
        self.__train_val_split = [0.7, 1]               # split of training and validaiton set (overlapping)
       
                                                        # Test set has to be small -> no overlap between it and other sets
        self.batch_size = batch_size

        if norm:
           self.norm = norm.lower()
        else:
            self.norm = norm

        # Set default File Paths if not given
        if config_file_path is None:
            config_file_path = Path(os.getcwd()).joinpath("config.ini")
        self.config_file = config_file_path


        # to be created and filled with values
        self.x_train:np.ndarray 
        self.y_train:np.ndarray 
        self.x_val:np.ndarray 
        self.y_val:np.ndarray
        self.x_test:np.ndarray 
        self.y_test:np.ndarray 

        # to remeber the last datetime of each Input Sample (needed if you want to log predictions to DB)
        self.train_times:list[datetime]
        self.val_times:list[datetime]
        self.test_times:list[datetime]

        # --------------- Initialize Dataset from Influx DB --------------
        
        # get dataframes containing all data
        # combine all keys to be loaded
        keys_to_load = self.input_keys.copy()

        for key,value in self.output_key.items():
            keys_to_load[key] = value
        
        
        # get the data from InfluxDB
        df_data = self.load_data(keys_to_load)

        # create datasets from data (make self.x_train, self.y_train, etc.)
        self.preprocess_data(df_data)

        logging.info(f"------ Finished Data preprocessing ------")


    def load_new_data(self,end_date:datetime,start_data:datetime)->tuple[np.ndarray,list[datetime]]:
        """
        Load new Input Data (for new predictions)

        Parameters:
            end_date (datetime):    Last Timestep (of Input)
            horizon (datetime):     First Timestep
        Returns:
            x_train (np.ndarray):   Input for new Predictions
        """

        logging.info(f"Starting with getting new data between [{start_data.date()} ... {end_date.date()}]")
        # HACK remeber old values
        old_start, old_end = self.start_date, self.end_date
        old_stride, old_output_window = self.stride, self.output_window

        # HACK set new values (for this new read)
        self.start_date, self.end_date = start_data, end_date
        self.stride = self.input_window

        # get new data
        df = self.load_data(self.input_keys)

        # normalize it with old normalizer
        df = self.__normalize_data(df, save_norm=False)

        ## Slice Data (only input data)
        # slice data into correct slices (for both x and y)
        slices_x = []           # all slices for model inputs
        end_times = []
        
        # how many samples (input data)
        for start in range(0, len(df), self.input_window):    # steps = 1 input window = 1 day
            
            # how much data points in a slice
            end_slice_x = start + self.input_window

            x_cols = list(self.input_keys.keys())
            df_slice = df[x_cols].iloc[start:end_slice_x]
            
            # data points
            slice_x = df_slice.values
            
            # times of the slice
            end_time = df_slice.index.max()

            # save the slices 
            slices_x.append(slice_x)
            end_times.append(end_time)

        # convert into numpy array
        x_data = np.array(slices_x)

        logging.info(f"Got new Data of Shape {x_data.shape}")

        # HACK reset old values
        self.start_date, self.end_date = old_start, old_end
        self.stride = old_stride
        
        return x_data, end_times

    

    def load_data(self, keys_to_load:dict) -> pd.DataFrame:
        """ 
        Load Data from Influx DB for all given keys
            - Merge into one Dataframe
            - Do some preprocessing (shift forecasts output_window back)
            - return Values (Numpy array) of DataFrame
            Args:
                keys_to_load (dict):    which keys (input AND output) shall be loaded
            Returns:
                df (pd.DataFrame):      All Values from the InfluxDB as 1 DataFrame (keys are column names)
        """

        # input data
        input_data = []

        for key,info in keys_to_load.items():

            db_reader = DfReader(bucket=info["bucket"], config_path= self.config_file)

            start_ts = int(self.start_date.timestamp())
            stop_ts = int(self.end_date.timestamp())

            # if the input was Weather Data -> consider it as a forecast -> get 1 more day
            if self.input_keys[key]["bucket"] == "weather":
                stop_ts = int((self.end_date + timedelta(hours=int(self.output_window/4))).timestamp())

            # load the dataframes
            df = db_reader.read(range_start=start_ts,
                           range_stop=stop_ts,
                           measurement_name=info["measurement"],
                           field_columns=[key])

            
            # resample df to 15min values and take neares values
            # df = df.resample('15T').nearest() # old resample version
            df = df.resample('15min').nearest()

            df.index = pd.to_datetime(df.index)
            # Remove timezone information (if any) and set to UTC for the index
            df.index = df.index.tz_localize(None).tz_localize('UTC')

            # if the input was Weather Data -> consider it as a forecast -> shift index back
            if self.input_keys[key]["bucket"] == "weather":
                df.index -+ timedelta(hours=int(self.output_window/4))      # shift it back by output window

            logging.info(f"\t Loaded {len(df)} values for key {key}")

            input_data.append(df[[key]].copy())

        # merge all the data sources together to 1 dataframe
        df_merged = pd.concat(input_data, axis = 1)
        
        # drop NaN values (so no "empty" entries are in the DF)
        df_merged = df_merged.dropna(how="any", axis = 0)

        # Identify missing intervals and reindex the DataFrame (so any cuts from the middle are reset)
        full_index = pd.date_range(start=df_merged.index.min(), end=df_merged.index.max(), freq=pd.Timedelta(minutes=15))
        df_reindexed = df_merged.reindex(full_index)

        # Interpolate these internal missing values
        df_interpolated = df_reindexed.interpolate(method='time')

        logging.info(f"\t Merged all Values into Dataset whith shape {df_interpolated.shape}")
        
        return df_interpolated

    def preprocess_data(self, df:pd.DataFrame)->None:
        """
        Preprocess the merged DataFrame delivered by load_data() to create 3 datasets (train/val/tes)
            1. Normalizes Data
                - Save the Mean and Std -> for de_normalisation()
            2. creates Slices of correct Size over the entire dataset 
                - save them as self.x_train ... 
                - x_Size: Input_window x Input_dimensions
                - y_Size: Output_window x 1
            3. Splits Data into 3 Datasets   
            4. Shuffle Datasets
        """
       
        # normalize data (i know small data leak -> but whatever man)
        df = self.__normalize_data(df)

        # slice data
        slices_x, slices_y, end_times = self.slice_data(df)

        # create train / val / test buckets
        self.__create_train_val_test_split(slices_x,slices_y, end_times)

        # shuffle Datasets
        self.__shuffle_datasets()

       
    def de_normalize(self,array:np.ndarray) -> np.ndarray:
        """
        Denormalize a Slice of Data 
            Only usable if data has been normalized beforehand

        Parameters:
            array (np.ndarray)  : Array to be denormalized (based on saved norm values)
        
        Returns:
            denormalized array
        """
        if self.norm:

            if self.norm == "std":
                # select correct mean values
                columns = list(self.output_key.keys())
                
                if array.shape[1] == len(self.input_keys):
                    columns = list(self.input_keys.keys())

                # read out from saved mean and std values    
                means = self.means[columns].values              # either vector (input) or scalar (target)
                std = self.stds[columns].values                 # either vector (input) or scalar (target)

                return ((array * std) + means)
            
            if self.norm == "minmax":
                raise NotImplementedError("minmax not yet implemented, use std or None")
            
        # no norm 
        return array
    

    def slice_data(self, df:pd.DataFrame)->tuple[list[np.ndarray],list[np.ndarray], list[datetime]]:
        """ 
        Slice the dataset into equal slices of length input_window (for X) and output_window (for Y)

        Parameters:
            df (pd.DataFrame)                         : DataFrame holding ALL data (from load_data())
        
        Returns:
            slices_x,slices_y,end_times (tuple[list[np.ndarray]]): List of all model Input and Target Slices and their end times
        
        """

        # slice data into correct slices (for both x and y)
        slices_x = []           # all slices for model inputs
        slices_y = []           # all slices for model output
        end_times = []
        for start in range(0, len(df) - (self.input_window + self.output_window), self.stride):
            
            end_slice_x = start + self.input_window

            # target is 1 sample ahead till output_wiwndow + 1 units ahead
            start_slice_y = end_slice_x + 1
            end_slice_y = start_slice_y + self.output_window 

            # select correct values and return them as numpy array
            x_cols = list(self.input_keys.keys())
            y_cols = list(self.output_key.keys())

            # get slice from start_x ... end_x
            slice_x = df[x_cols].iloc[start:end_slice_x].values

            # get slice from start_y ... end_y 
            slice_y = df[y_cols].iloc[start_slice_y:end_slice_y].values     # only output key for target                    
           


            # # empty numpy array to be filled (for one prediction)
            # data_slice = np.zeros(shape = (self.input_window, self.input_dim))
            
            # # fill the array with correct values from DF
            # for col_nr, col in enumerate(df):
                
            #     # forecasts go until output_window + input_window
            #     if self.input_keys[col]["bucket"] == "weather":
            #         # if so take values up till output time
            #         data_slice[:,col_nr] = df[col].iloc[(end_slice_y-self.input_window):end_slice_y].values
            #     else:
            #         # historical values only till input_window
            #         data_slice[:,col_nr] = df[col].iloc[start:end_slice_x].values

            # data_x = data_slice

            # also read out the endtime of the (historic) input values to save them 
            end_time = df.iloc[start:end_slice_x].index.max()
                       
            slices_x.append(slice_x)
            slices_y.append(slice_y)
            end_times.append(end_time)

        if len(slices_y) != len(slices_x):
            raise ValueError(f"Error during Slicing of dataset: X: has {len(slices_x)} and Y: has {len(slices_y)} slices")

        logging.info(f"Created {len(slices_x)} slices of data")
        
        return slices_x, slices_y, end_times
    
    def __create_train_val_test_split(self, slices_x:list[pd.Series], slices_y:list[pd.Series], end_times:list[datetime]):
        """ 
        Select the non - overlapping test and train slices from the dataset
        Then split train into train & validation set

        Parameters:
            slices_x: Slices for Model Inputs
            slices_y: Sliced for Model Outputs
        
        Returns:
            All split datasets (x_train, x_val, x_test, y_train, y_val, y_test)
        """

        nr_of_samples = len(slices_x) - self.input_window - self.output_window - 1
    
        # select TEST set first, calculate which indicies are now now valid for train and validation set
        # as there is an overlap between slices
        test_indices = np.random.choice(nr_of_samples, self.test_samples, replace = False)
        no_go_zone = set()
        
        # first create empy list to be filled and converted
        self.x_train = []
        self.y_train = []

        self.x_val = []
        self.y_val = []

        self.x_test = []
        self.y_test = []

        self.val_times = []
        self.train_times = []
        self.test_times = []
        

        # first fill test set
        for idx in test_indices:
            # fill test set
            self.x_test.append(slices_x[idx])
            self.y_test.append(slices_y[idx])
            self.test_times.append(end_times[idx])

            # calculate no-go-area (where test-set-samples are)
            for idx_range in range(idx, idx+self.input_window+self.output_window):
                no_go_zone.add(idx_range)
        
        # slice the rest of the dataset (without any test overlap)
        for idx in range(0, nr_of_samples):
            
            # skip idcs in test set
            if idx in no_go_zone:
                continue
            
            # calculate if the next sample is a train or validation sample
            # if random between [0 ... 0.99] < 0.7 --> train sample
            is_train = self.__train_val_split[0] > self.__train_val_split[1] * np.random.random()

            if is_train:
                self.x_train.append(slices_x[idx])
                self.y_train.append(slices_y[idx])
                self.train_times.append(end_times[idx])
            else:
                self.x_val.append(slices_x[idx])
                self.y_val.append(slices_y[idx])
                self.val_times.append(end_times[idx])

        # now calculate actual split ratio (of selection without overlap)
        nr_of_samples = len(self.x_train) + len(self.x_val) + len(self.x_test)
        # bool list of all test indices
        # self.test_idcs = [True if idx in no_go_zone else False for idx in range(len(slices_x))]
        self.is_test_idcs = np.array([0.0 for _ in range(len(slices_x))])
        self.is_test_idcs[list(no_go_zone)] += 1.0

        # calculate actual train, val, test split
        self.train_test_split = [len(self.x_train)/nr_of_samples, (len(self.x_train) + len(self.x_val))/nr_of_samples,1.00]
        self.train_test_split = [round(elem,3) for elem in self.train_test_split]

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_val = np.array(self.x_val)
        self.y_val = np.array(self.y_val)
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)

        
        logging.info(f"Created Train, Val, Test Samples of Size:       {[len(self.x_train),len(self.x_val),len(self.x_test)]}")
        logging.info(f"This is an actual train,val,test split ratio of {self.train_test_split}")

    def __shuffle_datasets(self):
        """ 
        Shuffle all created datasets 
            - shuffles x_train and y_train (same shuffel)
            - shuffles x_val and y_val (same shuffel)
            - shuffel x_test and y_test (same shuffel)
        """   

        # randomly permutate sequence of indices of x_train (default is 0 ... len(x_train))
        prem_train = np.random.choice(len(self.x_train), len(self.x_train), replace=False)

        # Apply the permutation to both arrays 
        self.x_train = self.x_train[prem_train] 
        self.y_train = self.y_train[prem_train] 
        # also permutate the times the same
        self.train_times = [self.train_times[idx] for idx in prem_train]

        # Generate a permutation of row indices (validation)
        prem_val = np.random.choice(len(self.x_val), len(self.x_val), replace=False)

        # Apply the permutation to both arrays 
        self.x_val = self.x_val[prem_val] 
        self.y_val = self.y_val[prem_val] 
        # also permutate the times the same
        self.val_times = [self.val_times[idx] for idx in prem_val]

        # Generate a permutation of row indices (test)
        prem_test = np.random.choice(len(self.x_test), len(self.x_test), replace=False)

        # Apply the permutation to both arrays 
        self.x_test = self.x_test[prem_test] 
        self.y_test = self.y_test[prem_test]
        # also permutate the times the same
        self.test_times = [self.test_times[idx] for idx in prem_test]


        logging.info(f"Shuffled all Datasets")


    def __normalize_data(self, df:pd.DataFrame, save_norm:bool=True) -> pd.DataFrame:
        """
        Normalize the DataFrame (standardize)

        Parameters:
            df (pd.DataFrame):      Entire Dataset (from load_data())
            norm (str):             Normalisation type ("std", "minmax", None)
            save_norm (bool):       Wether to Save Dataset info
        Returns:
            df_norm (pd.DataFrame): Normalized Dataset
        """

        if self.norm:

            if self.norm == "std":
            
                # save the values (for de_normalisation)
                if save_norm:
                    self.means = df.mean()
                    self.stds = df.std()

                df_norm = (df - self.means) / self.stds

                return df_norm
            
            if self.norm == "minmax":
                raise NotImplementedError("MinMax Norm Not yet implemented, use std or None")
        
        # for everything else -> no norm
        return df
    
    def print_shape_info(self):
        """
        Prints the Shape information of all Datasets (if logging level == info)
        """

        # get formated data
        logging.info("Created Dataset have the following Shapes:")
        logging.info(f"\t x_train : {self.x_train.shape}")
        logging.info(f"\t y_train : {self.y_train.shape}")
        logging.info(f"\t x_val   : {self.x_val.shape}")
        logging.info(f"\t y_val   : {self.y_val.shape}")
        logging.info(f"\t x_test  : {self.x_test.shape}")
        logging.info(f"\t y_test  : {self.y_test.shape}")

if __name__ == "__main__":

    # Set above INFO (WARNING, etc) to disable all the info logs
    logging.basicConfig(level=logging.INFO)

    # -------- EDIT THIS PART IF YOU WANT TO USE IT -----------
    # See LookupTable of Meteo Logger and Aawasser Logger for this info
    # or go checkout Influx DB
    input_keys = {
        "PV-Anlage":{
            "bucket":"hist",
            "measurement":"power_kW"
        },
        "shortwave_radiation_instant":{
            "bucket":"weather",
            "measurement":"irradiance"
        }
    }

    output_key = {
        "PV-Anlage":{
            "bucket":"hist",
            "measurement":"power_kW"
        }
    }
    
    # change date for more / less data
    start_date = datetime(year=2023, month=10, day=10)
    end_date = datetime(year=2024, month=2, day=6)


    # ----------------------------

    dataset = Dataset(input_keys= input_keys,
                      input_window= 24,
                      output_key= output_key,
                      output_window=24,
                      start_date=start_date,
                      end_date=end_date)

    # print the shapes of all datasets
    dataset.print_shape_info()
    
    # if you only want 1 (new) sample of model input data
    end_2 = datetime(year=2024, month=8, day=5) 
    start_2 = end_2 - timedelta(hours=int(24/4))
    

    x_data = dataset.load_new_data(end_date=end_2,start_data=start_2)

    # --------- some prints ------------
    import matplotlib.pyplot as plt

    plt.plot(dataset.is_test_idcs)
    plt.xlabel(f"Index")
    plt.ylabel(f"Is in Test Set")
    plt.title(f"Test Samples Selection")
    plt.show()
