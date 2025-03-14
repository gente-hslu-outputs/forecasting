from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import os

import numpy as np
import pandas as pd

from abc import ABC,abstractmethod

from keras.models import load_model

# GitLab Imports
from databaseaccessor import DfWriter, DfReader
from forecaster import create_simple_dense, create_simple_cnn, create_simple_tcn
from forecaster import Dataset


class Predictor:
    """
    Predictor with all functionalities

        On init is either set as Load_Predictor or PV_Predictor
    """
    
    def __init__(self, input_window:int = 24, output_window:int = 24, end_date:datetime=None, days_of_data:int = 365, config_file_path:str = None) -> None:
        """
        Init a predictor 
        
        """
        self.input_window = input_window
        self.output_window = output_window
        
        if end_date is None:
            self.end_date = datetime.now(timezone.utc) - timedelta(days = 2)
        else: 
            self.end_date = end_date

        self.start_date = self.end_date.replace(tzinfo=timezone.utc) - timedelta(days=days_of_data)


        # get a Dataset
        self.dataset = Dataset(input_keys=self.input_keys,
                               input_window=self.input_window,
                               output_key=self.output_key,
                               output_window=self.output_window,
                               start_date=self.start_date,
                               end_date=self.end_date,
                               norm = None,                  # Normalisaton function (std, minmax, None)
                               config_file_path=config_file_path)
        
        self.dataset.print_shape_info()
  

        # Set default File Paths if not given
        if config_file_path is None:
            config_file_path = Path(os.getcwd()).joinpath("config.ini")
        self.config_file = config_file_path

        # initialize a DB Writer (is always the same)
        self.db_writer =  DfWriter(bucket = "pred",config_path = self.config_file)


    def train(self, epochs:int=100, steps_per_epoch:int = None, val_steps:int = None):
        """
        Train the Model
        """

        logging.info(f"Start Model Training")

        history = self.model.fit(x=self.dataset.x_train,
                       y=self.dataset.y_train,
                       batch_size=self.dataset.batch_size,
                       epochs = epochs,
                       steps_per_epoch = steps_per_epoch,
                       validation_data=(self.dataset.x_val,self.dataset.y_val),
                       validation_steps = val_steps)
        
        for key,values in history.history.items():
            logging.info(f"Smallest {key}: {np.min(values)}")

    def post_process(self,y_pred:np.ndarray,times:list[datetime])->np.ndarray:
        """
        Post Process any Prediction (from PV) to be 0 at night
        """

        # clip some values
        if self.dataset.norm == None:
                
                # no negative values (if not normalized)
                y_pred[y_pred<0] = 0

                # no load / production above 200 kW
                y_pred[y_pred>200] = 0        


        # no special post process (for Non PV)
        if type(self) != Predictor_PV:
            
            return y_pred
        
        # Set values to 0 when it is night
        else:
            # setup reader to read from DB
            db_reader = DfReader(bucket="weather",config_path= self.config_file)

            # go though all samples
            for idx, end_time in enumerate(times):
                
                # calculate when the prediction was and convert to timestamp
                start_ts = int(end_time.timestamp())
                end_ts = int((end_time + timedelta(hours=self.output_window//4)).timestamp())
                
                # get daytime values from prediction
                df_day = db_reader.read(range_start=start_ts,
                                        range_stop=end_ts,
                                        measurement_name="bool",
                                        field_columns=["is_day"])

                # mask with is_day 
                y_pred[idx,:] = df_day["is_day"].values.astype(float) * y_pred[idx,:]

                return y_pred
        
    
    def predict_new(self, end_date:datetime, horizon:timedelta, log_pred:bool = True)->np.ndarray:
        """ 
        Make a prediction on new data
            1. load this new data
            2. make prediction on it
            3. log the prediction (if enabled)

        Parameters:
            end_date (datetime):    Last Timestep (of Input Data)
            horizon (timedelta):    How long to log data for (>= input_window)
            lag_values (bool):      If to log results to DataBase

        Returns:
            y_pred (ndarray):       Prediction
        """
        
        start_date = end_date - horizon

        logging.info(f"Start a new prediction for [{end_date.date()} ... {(end_date+timedelta(minutes=15 * (1+self.output_window))).date()}]")
        
        x_data, times = self.dataset.load_new_data(end_date = end_date,
                                                   start_data = start_date)

        if len(x_data.shape) == 2:
            # Reshape to (n, 96, 1) by adding a new axis
            x_data = x_data[:, :, np.newaxis] 

        y_pred = self.model.predict(x_data)

        # reshape into shape (#samples,output_window)
        y_pred = y_pred.reshape((x_data.shape[0],self.output_window))

        logging.info(f"Got new predictions of shape {y_pred.shape}")

        # post processing prediction

        y_pred = self.post_process(y_pred=y_pred,times=times)
        
       
        if log_pred:
            for i in range(y_pred.shape[0]):
                logging.info(f"Logging new prediction sample {i} from {times[i].date()} to {list(self.output_key.keys())[0]} in pred bucket of DB")
                self.log(y_pred=y_pred[i], end_time = times[i])

        
        return y_pred
        # reset old values

    def predict_dataset(self,dataset_kind:str = "train")->np.ndarray:
        """
        Predict ALL values from a downloaded dataset 
            and maybe log the results in the DB
        
        Parameters:
            dataset_kind (str): Optional: What kind of Dataset to use ("train", "val", or "test") defaults to "train"
            log_pred (bool):    Optional: Wether to log values or not, defaults to False
        """
        logging.info(f"Predicting entire {dataset_kind.upper()} dataset")
        # select correct Data and EndTimes List
        x_data = self.dataset.x_train
        times = self.dataset.train_times
        if dataset_kind.lower() == "val":
            x_data = self.dataset.x_val
            times = self.dataset.val_times
        if dataset_kind.lower() == "test":
            x_data = self.dataset.x_test
            times = self.dataset.test_times

        predictions = self.model.predict(x_data)
        
        predictions = self.post_process(predictions, times)
       
        return predictions

    def predict_sample(self, sample_nr:int=0, dataset_kind:str = "train", log_pred:bool=False)->np.ndarray:
        """
        Predict one Sample from a downloaded dataset 
            and maybe log the results in the DB
        
        Parameters:
            sample_nr (int):    Optional: Which sample to take, defaults to 0
            dataset_kind (str): Optional: What kind of Dataset to use ("train", "val", or "test") defaults to "train"
            log_pred (bool):    Optional: Wether to log values or not, defaults to False
        """
        logging.info(f"Predicting ONE {dataset_kind.upper()} Sample: [{sample_nr}]")

        # select correct Data and EndTimes List
        x_data = self.dataset.x_train[sample_nr,:,:]
        end_time = self.dataset.train_times[sample_nr]
        if dataset_kind.lower() == "val":
            x_data = self.dataset.x_val[sample_nr,:,:]
            end_time = self.dataset.val_times[sample_nr]
        if dataset_kind.lower() == "test":
            x_data = self.dataset.x_test[sample_nr,:,:]
            end_time = self.dataset.test_times[sample_nr]

        # add the 1st axis back (as to have 3D Data)

        x_data = x_data[np.newaxis,:,:]

        y_pred = self.model.predict(x_data)

        # post process data (set night of pv to 0)
        y_pred = self.post_process(y_pred=y_pred, times = [end_time])
        
        if log_pred:
            logging.info(f"Logging Sample {sample_nr} from {end_time.date()} to {list(self.output_key.keys())[0]} in pred bucket of DB")
            self.log(y_pred, end_time)

        return y_pred
    

    def test(self)->tuple[np.ndarray,np.ndarray]:
        """
        Test the Model with a the test set

        Returns:
            (y_pred, y_true) (tuple[np.ndarray]):    Prediction and Target
        """

        y_pred = self.model.predict(x=self.dataset.x_test)
        y_true = self.dataset.y_test

        # reshape y_pred if necessary
        y_pred = y_pred.reshape(y_true.shape)

        logging.info(f"Predictor has an Mean Absolute Test Error of {np.mean(np.abs(y_pred - y_true)):.4f}")
        
        return y_pred, y_true


    def load_model(self,model_path:Path):
        """ 
        Load a saved Model (as self.model)
        """

        self.model = load_model(filepath=model_path)
        
        logging.info(f"Loaded the following model:")
        self.model.summary()

        self.model.compile(optimizer = "Adam",
                           loss = "mse",
                           metrics = ["mae"])
        

    def log(self,y_pred:np.ndarray, end_time:datetime, de_norm:bool = True):
        """
        Log the prediction into the Database

        Parameters:
            y_pred (np.ndarray):    Model Predictions
            end_time (datetime):    Last timestep of the Input Data
            de_norm (bool):         If to denormalize the data (defaults to true)
        """
        y_pred = y_pred.reshape((self.output_window,1))       # make them be a column vector (for de - norm)

        if de_norm:
            # de-normalize data before writing
            y_pred = self.dataset.de_normalize(y_pred)

        # flatten again (for dataframe)
        y_pred = y_pred.flatten()

        # create new DataFrame with the Predictions in it
        # adjust timeframe from end of input to output
        # output is in timeframe [input_window + 1 .... input_window + output_window + 1]
        idcs = [end_time + (i * timedelta(minutes = 15)) for i in range(1, self.output_window+1)]
        pred_name = list(self.output_key.keys())[0]
        df_to_log = pd.DataFrame({pred_name:y_pred}, index = idcs)
        
        measurement_name = self.output_key[pred_name]["measurement"]
        
        # log the DataFrame
        self.db_writer.write(df=df_to_log, measurement_name=measurement_name)
        


class Predictor_Load(Predictor):
    """ 
    Load Predictor with all HardCoded changes in __init__
    """
    def __init__(self, input_window:int = 24, output_window:int = 24,end_date:datetime=None, days_of_data:int = 365, config_file_path:str = None) -> None:
        """ 
        Create a Load Predictor

        Parameters:
            input_window (int)      :    How many timesteps for Model Input
            output_window (int)     :    How many timesteps for Model Output
            days_of_data (int)      :    How much data to load (for training)
            config_file_path (str)  :    Path to config.ini file for dbwriter
        """

        logging.info(F"Create LOAD Predictor")

        # -------- HARD CODED --------------
        # change these keys for other inputs
        # ----------------------------------
        self.input_keys = {
            "Total Load":{
                "bucket":"hist",
                "measurement":"power_kW"
            },
        }

        # defines TARGET
        # is stored in DB (pred bucket)
        self.output_key = {
            "Total Load":{
                "bucket":"hist",
                "measurement":"power_kW"
            },
        }

        # load the correct model
        self.model = create_simple_tcn(input_window=input_window,
                                        input_dim=len(self.input_keys),
                                        output_window=output_window)
        
        self.model.summary()

        self.model.compile(optimizer = "Adam",
                           loss = "mse",
                           metrics = ["mae"])         

        super().__init__(input_window=input_window,
                         output_window=output_window,
                         end_date = end_date,
                         days_of_data= days_of_data,
                         config_file_path= config_file_path)

               


class Predictor_PV(Predictor):
    """ 
    PV Predictor with all HardCoded changes in __init__
    """

    def __init__(self, input_window:int = 24, output_window:int = 24,end_date:datetime=None, days_of_data:int = 365, config_file_path:str = None) -> None:
        """ 
        Create a PV Predictor

        Parameters:
            input_window (int)      :    How many timesteps for Model Input
            output_window (int)     :    How many timesteps for Model Output
            days_of_data (int)      :    How much data to load (for training)
            config_file_path (str)  :    Path to config.ini file for dbwriter
        """

        logging.info(F"Create PV Predictor")

        # get db_reader (for is_day)
        self.db_reader = DfReader(bucket  = "weather", config_path=config_file_path)

        # -------- HARD CODED --------------
        # change these keys for other inputs
        # ----------------------------------
        # self.input_keys = {
        #     "PV-Anlage":{
        #         "bucket":"hist",
        #         "measurement":"power_kW"
        #     },
        #     "shortwave_radiation_instant":{
        #         "bucket":"weather",
        #         "measurement":"irradiance"
        #     },
        #     "cloud_cover":{
        #         "bucket":"weather",
        #         "measurement":"percent"
        #     }
        # }

        self.input_keys = {
            "PV-Anlage":{
                "bucket":"hist",
                "measurement":"power_kW"
            },
            "shortwave_radiation_instant":{
                "bucket":"weather",
                "measurement":"irradiance"
            },
        }

        # defines TARGET
        # is stored in DB (pred bucket)
        self.output_key = {
            "PV-Anlage":{
                "bucket":"hist",
                "measurement":"power_kW"
            },
        }

        #load the correct model
        self.model = create_simple_cnn(input_window=input_window,
                                        input_dim=len(self.input_keys),
                                        output_window=output_window)   
        # self.model = create_simple_tcn(input_window=input_window,
        #                                 input_dim=len(self.input_keys),
        #                                 output_window=output_window)  
        self.model.summary()

        self.model.compile(optimizer = "Adam",
                           loss = "mse",
                           metrics = ["mae"])

        super().__init__(input_window=input_window,
                         output_window=output_window,
                         end_date = end_date,
                         days_of_data= days_of_data,
                         config_file_path= config_file_path)
        


if __name__ == "__main__":

    # if set to INFO -> will print out information about proceedings
    logging.basicConfig(level=logging.INFO)


    import os
    from pathlib import Path
    config_file = Path(os.getcwd()).joinpath("config.ini")

    
    # select last day to get data (for training)
    end_date = datetime(year=2024,month=8, day = 10, hour=00, minute=00)

    # Choose Predictor
    predictor = Predictor_PV(end_date = end_date,
                             days_of_data=365,
                             config_file_path=config_file)
    

    # predictor = Predictor_Load()

    # Train the predictor
    # predictor.train(epochs = 1)

    # save the trained model
    # predictor.model.save("/models/model_simple")

    # load the model (if you want)
    # predictor.load_model("/models/model_simple")

    # # make a prediction and store the sample
    # y_pred = predictor.predict_sample(log_pred=True)

    # # test the predictor
    # predictor.test()

    # # predict ALL values from any given dataset and log the values
    # predictor.predict_dataset(dataset_kind="test", log_pred=True)


    # or if you want to just predict on some new data
    # by loading this and then predicting on it
    end_date = datetime.now() - timedelta(hours=1)
    horizon = timedelta(hours = 24)

    predictor.predict_new(end_date = end_date, 
                          horizon= horizon,
                          log_pred = True)