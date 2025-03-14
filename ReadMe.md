# Forecaster Library

## Table of Contents
- [Forecaster Library](#forecaster-library)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Local Installation](#local-installation)
    - [Installation via GitLab](#installation-via-gitlab)
  - [Usage](#usage)
    - [Logging the Predictions](#logging-the-predictions)


## Installation

### Local Installation

Clone this Repo and `cd` into this folder.
Then install this library with:

```
pip install .
```

### Installation via GitLab

Requires that you have a `.pypirc` File setup, and that you know:
- `username` (is `__token__` in the command)
- `password` (is `<your_personal_token>` in the command below)

```
pip install forecaster --index-url https://__token__:<your_personal_token>@gitlab.com/api/v4/projects/45647566/packages/pypi/simple
```


## Usage

This Library offerst two Predictors. 

```python
from forecaster import Predictor_PV
from forecaster import Predictor_Load
```

Each predictor first loads data from the Indlux DB,
as to prepare its datasets. (during `__init__`)
The amount of data can be set during creation.

```python
predictor = Predictor_PV(days_of_data = 365)
```
Then each predictor has a model which can be:
- trained
- saved
- loaded
- used for predictions
  - on individual samples
  - on entire datasets
  - on new (unknown) data

```python
predictor.train(epochs = 1)

predictor.model.save("/models/model_simple")

# load a saved model (if you want)
predictor.load_model("/models/model_simple")

# test the predictor (on the test set)
predictor.test()

# ----------------
# Make Predictions
# ----------------

# make a prediction from the loaded data and store the sample
y_pred = predictor.predict_sample(log_pred=True)

# predict ALL values from any loaded dataset and log the values
predictor.predict_dataset(dataset_kind="test", log_pred=True)

# or if you want to just predict on some new data
# this loads new data from the DataBase
end_date = datetime.now() - timedelta(hours=1)
horizon = timedelta(hours = 24)

predictor.predict_new(end_date = end_date, 
                        horizon= horizon,
                        log_pred = True)
```

### Logging the Predictions

Each `predict` Method has the attribute `log_pred`, if set to `True` the predictions will be `de_normalized` and saved to the DataBase under the `pred` bucket.

```python
# un-saved prediction
y_pred = predictor.predict_sample()

# saved prediction
y_pred = predictor.predict_sample(log_pred=True)
```
