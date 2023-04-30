from datetime import datetime
import os
import json

import pandas as pd

import pickle
import joblib

from flask import Flask, request, jsonify
from peewee import (
    Model, BooleanField, FloatField,
    IntegerField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

########################################
# Begin constants

THRESHOLD_LGBM = 0.16602175975807876

# End constants
########################################

########################################
# Begin database

# DB = SqliteDatabase('predictions.db')
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')


class Prediction(Model):
    observation = TextField()
    observation_id = TextField(unique=True)
    probability = FloatField()
    predicted_outcome = BooleanField()
    true_outcome = BooleanField(null=True)
    type = TextField()
    date = TextField()
    part_of_a_policing_operation = BooleanField(null=True)
    latitude = FloatField(null=True)
    longitude = FloatField(null=True)
    gender = TextField()
    age_range = TextField()
    officer_defined_ethnicity = TextField()
    legislation = TextField()
    object_of_search = TextField()
    station = TextField()
    hour = IntegerField()
    day_of_week = TextField()
    month = IntegerField()

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database
########################################

########################################
# Unpickle the previously-trained model

with open('columns.json') as fh:
    columns = json.load(fh)

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

pipeline = joblib.load('pipeline.pickle')

# End model un-pickling
########################################

########################################
# Input validation functions


def check_request(observation):
    """
        Validates that our request is well formatted

        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, error message otherwise
    """

    if "observation_id" not in observation:
        error = f"Field `observation_id` missing from observation: {observation}"
        return False, error

    return True, ""


def check_valid_column(observation):
    """
        Validates that our observation only has valid columns

        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, error message otherwise
    """

    valid_columns = {
        "observation_id",
        "Type",
        "Date",
        "Part of a policing operation",
        "Latitude",
        "Longitude",
        "Gender",
        "Age range",
        "Officer-defined ethnicity",
        "Legislation",
        "Object of search",
        "station"
    }

    keys = set(observation.keys())

    if len(valid_columns - keys) > 0:
        missing = valid_columns - keys
        error = f"Missing columns: {missing}"
        return False, error

    if len(keys - valid_columns) > 0:
        extra = keys - valid_columns
        error = f"Unrecognized columns provided: {extra}"
        return False, error

    return True, ""


def check_categorical_values(observation):
    """
        Validates that all categorical fields are in the observation and values are valid

        Returns:
        - assertion value: True if all provided categorical columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, error message otherwise
    """

    valid_category_map = {
        "Type": ['Person search', 'Person and Vehicle search', 'Vehicle search'],
        # "Part of a policing operation": [False, True],
        "Gender": ['Male', 'Female', 'Other'],
        "Age range": ['18-24', '25-34', 'over 34', '10-17', 'under 10'],
        "Officer-defined ethnicity": ['White', 'Black', 'Asian', 'Mixed', 'Other'],
        "Legislation": [
            'Police and Criminal Evidence Act 1984 (section 1)',
            'Misuse of Drugs Act 1971 (section 23)',
            'Firearms Act 1968 (section 47)',
            'Poaching Prevention Act 1862 (section 2)',
            'Psychoactive Substances Act 2016 (s36(2))',
            'Criminal Justice and Public Order Act 1994 (section 60)',
            'Criminal Justice Act 1988 (section 139B)',
            'Psychoactive Substances Act 2016 (s37(2))',
            'Wildlife and Countryside Act 1981 (section 19)',
            'Aviation Security Act 1982 (section 27(1))',
            'Public Stores Act 1875 (section 6)',
            'Hunting Act 2004 (section 8)',
            'Environmental Protection Act 1990 (section 34B )',
            'Customs and Excise Management Act 1979 (section 163)',
            'Police and Criminal Evidence Act 1984 (section 6)',
            'Conservation of Seals Act 1970 (section 4)',
            'Crossbows Act 1987 (section 4)',
            'Deer Act 1991 (section 12)',
            'Protection of Badgers Act 1992 (section 11)'
        ],
        "Object of search": [
            'Article for use in theft',
            'Controlled drugs',
            'Stolen goods',
            'Offensive weapons',
            'Anything to threaten or harm anyone',
            'Articles for use in criminal damage',
            'Firearms',
            'Game or poaching equipment',
            'Psychoactive substances',
            'Fireworks',
            'Evidence of offences under the Act',
            'Goods on which duty has not been paid etc.',
            'Detailed object of search unavailable',
            'Crossbows',
            'Evidence of wildlife offences',
            'Evidence of hunting any wild mammal with a dog',
            'Seals or hunting equipment'
        ],
        "station": [
            'metropolitan',
            'hampshire',
            'sussex',
            'south-yorkshire',
            'staffordshire',
            'bedfordshire',
            'cumbria',
            'nottinghamshire',
            'warwickshire',
            'devon-and-cornwall',
            'leicestershire',
            'durham',
            'humberside',
            'thames-valley',
            'norfolk',
            'lincolnshire',
            'northamptonshire',
            'kent',
            'wiltshire',
            'city-of-london',
            'west-mercia',
            'lancashire',
            'cheshire',
            'derbyshire',
            'btp',
            'surrey',
            'hertfordshire',
            'north-yorkshire',
            'cambridgeshire',
            'northumbria',
            'west-yorkshire',
            'north-wales',
            'cleveland',
            'gloucestershire',
            'avon-and-somerset',
            'dorset',
            'suffolk',
            'dyfed-powys',
            'west-midlands',
            'merseyside',
            'essex'
        ]
    }

    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ", ".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        else:
            error = f"Categorical field {key} missing"
            return False, error

    return True, ""


def check_coordinates(observation):
    """
        Validates that coordinates contain valid values

        Returns:
        - assertion value: True if both coordinates contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, error message otherwise
    """

    latitude = observation.get('Latitude')
    longitude = observation.get('Longitude')

    if latitude is None:
        error = "Field `Latitude` is missing"
        return False, error

    if longitude is None:
        error = "Field `Longitude` is missing"
        return False, error

    if not isinstance(latitude, float):
        error = "The Field `Latitude` is not a float"
        return False, error

    if not isinstance(longitude, float):
        error = "The Field `Longitude` is not a float"
        return False, error

    if latitude < -90. or latitude > 90.:
        error = f"The Field `Latitude` - value `{latitude}` is not between -90.0 and 90.0"
        return False, error

    if longitude < -90. or longitude > 90.:
        error = f"The Field `Longitude` - value `{longitude}` is not between -90.0 and 90.0"
        return False, error

    return True, ""


def check_date(observation):
    """
        Validates that coordinates contain valid values

        Returns:
        - assertion value: True if both coordinates contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, error message otherwise
    """

    date_string = observation.get('Date')

    if date_string is None:
        error = "Field `Date` missing"
        return False, error

    if not isinstance(date_string, str):
        error = "The Field `Date` is not a string"
        return False, error

    try:
        datetime.fromisoformat(date_string)
    except:
        error = "The Field `Date` is not in ISO8601 format"
        return False, error

    return True, ""


def check_observation(observation):
    """
        Validates that our observation has passed all checks

        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    request_ok, error = check_request(observation)
    if not request_ok:
        return False, error

    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        return False, error

    categories_ok, error = check_categorical_values(observation)
    if not categories_ok:
        return False, error

    coordinates_ok, error = check_coordinates(observation)
    if not coordinates_ok:
        return False, error

    date_ok, error = check_date(observation)
    if not date_ok:
        return False, error

    return True, ""


def check_result(observation):
    """
        Validates that our result is valid

        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """

    if "observation_id" not in observation:
        error = f"The Field `observation_id` missing from search result: {observation}"
        return False, error

    if not isinstance(observation.get('observation_id'), str):
        error = "The Field `observation_id` is not a string"
        return False, error

    if "outcome" not in observation:
        error = f"The Field `outcome` missing from search result: {observation}"
        return False, error

    if not isinstance(observation.get('outcome'), bool):
        error = "The Field `outcome` is not a boolean"
        return False, error

    return True, ""


def validate_part_of_a_policing_operation(raw_observation) -> bool:
    return raw_observation if isinstance(raw_observation, bool) else False


def get_observation_dataframe(raw_observation) -> pd.DataFrame:
    observation = {}
    observation['type'] = raw_observation['Type']
    observation['part_of_a_policing_operation'] = validate_part_of_a_policing_operation(
        raw_observation['Part of a policing operation'])
    observation['latitude'] = raw_observation['Latitude']
    observation['longitude'] = raw_observation['Longitude']
    observation['gender'] = raw_observation['Gender']
    observation['age_range'] = raw_observation['Age range']
    observation['officer_defined_ethnicity'] = raw_observation['Officer-defined ethnicity']
    observation['legislation'] = raw_observation['Legislation']
    observation['object_of_search'] = raw_observation['Object of search']
    observation['station'] = raw_observation['station']

    df_iso = pd.DataFrame({'date': raw_observation['Date']}, index=[0])
    df_iso.date = pd.to_datetime(df_iso.date, infer_datetime_format=True)
    df_iso['hour'] = df_iso.date.dt.hour
    df_iso['day_of_week'] = df_iso.date.dt.day_name()
    df_iso['month'] = df_iso.date.dt.month

    observation['hour'] = df_iso.hour.iloc[0]
    observation['day_of_week'] = df_iso.day_of_week.iloc[0]
    observation['month'] = df_iso.month.iloc[0]

    return pd.DataFrame([observation], columns=columns).astype(dtypes)

# End input validation functions
########################################

########################################
# Begin webserver


app = Flask(__name__)


@app.route('/should_search/', methods=['POST'])
def should_search():

    observation = request.get_json()

    observation_ok, error_msg = check_observation(observation)
    if not observation_ok:
        print(f"\n{error_msg}\n")
        error_response = {
            "error": error_msg
        }

        return error_response, 405

    obs = get_observation_dataframe(observation)
    probability = pipeline.predict_proba(obs)[0, 1]
    predicted_outcome = True if probability > THRESHOLD_LGBM else False

    print('part_of_a_policing_operation type:',
          type(bool(obs.part_of_a_policing_operation[0])))
    print('part_of_a_policing_operation:', obs.part_of_a_policing_operation[0])

    p = Prediction(
        observation=observation,
        observation_id=observation['observation_id'],
        probability=probability,
        predicted_outcome=predicted_outcome,
        type=obs.type[0],
        date=observation['Date'],
        part_of_a_policing_operation=bool(obs.part_of_a_policing_operation[0]),
        latitude=obs.latitude[0],
        longitude=obs.longitude[0],
        gender=obs.gender[0],
        age_range=obs.age_range[0],
        officer_defined_ethnicity=obs.officer_defined_ethnicity[0],
        legislation=obs.legislation[0],
        object_of_search=obs.object_of_search[0],
        station=obs.station[0],
        hour=obs.hour[0],
        day_of_week=obs.day_of_week[0],
        month=obs.month[0]
    )

    try:
        p.save()

        response = {'outcome': predicted_outcome}
        print(f"\n{response}\n")

        return response

    except IntegrityError as exception:
        DB.rollback()

        error_msg = f"Observation Id: \'{observation['observation_id']}\' already exists"
        print(f"\n{error_msg}\n")
        print(f"Peewee error message: {exception}\n")
        error_response = {
            "error": error_msg
        }

        return error_response, 405

    except Exception as exception:

        error_msg = f"Peewee error message: {exception}\n"
        print(f"\n{error_msg}\n")
        error_response = {
            "error": error_msg
        }

        return error_response, 405


@app.route('/search_result/', methods=['POST'])
def search_result():
    observation = request.get_json()

    result_ok, error_msg = check_result(observation)
    if not result_ok:
        print(f"\n{error_msg}\n")
        error_response = {
            "error": error_msg
        }

        return error_response, 405

    try:
        p = Prediction.get(Prediction.observation_id ==
                           observation['observation_id'])
        p.true_outcome = observation['outcome']
        p.save()

        response = {
            "observation_id": p.observation_id,
            "outcome": p.true_outcome,
            "predicted_outcome": p.predicted_outcome
        }
        print(f"\n{response}\n")

        return response

    except Prediction.DoesNotExist:
        error_msg = f"Observation Id: \'{observation['observation_id']}\' does not exist"
        error_response = {
            "error": error_msg
        }
        print(f"\n{error_msg}\n")
        return error_response, 405


@app.route('/db_list/')
def db_list():
    response = [
        model_to_dict(obs) for obs in Prediction.select()
    ]
    print(f"\n{response}\n")

    return response


@app.route('/delete/')
def delete():
    response = {'deleted_rows': Prediction.delete().execute()}
    print(f"\n{response}\n")

    return response

# End webserver
########################################


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
