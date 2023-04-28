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

SUCCESS_RATE = 0.10

# End constants
########################################

########################################
# Begin database

# DB = SqliteDatabase('predictions.db')
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')


class Prediction(Model):
    observation = TextField()
    proba = FloatField()
    predicted_outcome = BooleanField()
    true_outcome = BooleanField(null=True)
    observation_id = TextField(unique=True)
    type = TextField()
    date = TextField()
    part_of_a_policing_operation = BooleanField()
    latitude = FloatField()
    longitude = FloatField()
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
        "station",
        "hour",
        "day_of_week",
        "month"
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


def check_categories(observation):
    """
        Validates that all categorical fields are in the observation and values are valid

        Returns:
        - assertion value: True if all provided categorical columns contain valid values, 
                           False otherwise
        - error message: empty if all provided columns are valid, error message otherwise
    """

    valid_category_map = {
        "Type": ['Person search', 'Person and Vehicle search', 'Vehicle search'],
        "Part of a policing operation": [True, False],
        "Gender": ['Male', 'Female', 'Other'],
        "Age range": ['18-24', '25-34', 'over 34', '10-17', 'under 10'],
        "Officer-defined ethnicity": ['White', 'Black', 'Asian', 'Mixed', 'Other'],
        "Legislation": [
            'Misuse of Drugs Act 1971 (section 23)',
            'Police and Criminal Evidence Act 1984 (section 1)',
            'Criminal Justice and Public Order Act 1994 (section 60)',
            'Firearms Act 1968 (section 47)',
            'Criminal Justice Act 1988 (section 139B)',
            'Psychoactive Substances Act 2016 (s36(2))',
            'Poaching Prevention Act 1862 (section 2)',
            'Police and Criminal Evidence Act 1984 (section 6)',
            'Wildlife and Countryside Act 1981 (section 19)',
            'Environmental Protection Act 1990 (section 34B )',
            'Aviation Security Act 1982 (section 27(1))',
            'Deer Act 1991 (section 12)',
            'Customs and Excise Management Act 1979 (section 163)',
            'Crossbows Act 1987 (section 4)',
            'Hunting Act 2004 (section 8)',
            'Conservation of Seals Act 1970 (section 4)',
            'Protection of Badgers Act 1992 (section 11)',
            'Public Stores Act 1875 (section 6)',
            'Psychoactive Substances Act 2016 (s37(2))'
        ],
        "Object of search": [
            'Controlled drugs',
            'Offensive weapons',
            'Stolen goods',
            'Article for use in theft',
            'Evidence of offences under the Act',
            'Articles for use in criminal damage',
            'Anything to threaten or harm anyone',
            'Firearms',
            'Fireworks',
            'Psychoactive substances',
            'Game or poaching equipment',
            'Detailed object of search unavailable',
            'Goods on which duty has not been paid etc.',
            'Crossbows',
            'Evidence of wildlife offences',
            'Evidence of hunting any wild mammal with a dog',
            'Seals or hunting equipment'
        ],
        "station": [
            'metropolitan', 'merseyside', 'thames-valley', 'west-yorkshire', 'south-yorkshire',
            'hampshire', 'btp', 'kent', 'lancashire', 'hertfordshire',
            'avon-and-somerset', 'essex', 'sussex', 'devon-and-cornwall', 'surrey',
            'humberside', 'west-midlands', 'west-mercia', 'staffordshire', 'norfolk',
            'leicestershire', 'cheshire', 'northumbria', 'cleveland', 'nottinghamshire',
            'north-wales', 'suffolk', 'bedfordshire', 'lincolnshire', 'dyfed-powys',
            'city-of-london', 'northamptonshire', 'warwickshire', 'durham', 'north-yorkshire',
            'gloucestershire', 'derbyshire', 'cambridgeshire', 'cumbria', 'wiltshire',
            'dorset'
        ],
        "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
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

    if (not isinstance(latitude, float)) or (not isinstance(longitude, float)):
        error = "The Field `Latitude` or `Longitude` is not a float"
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

    dt_str = observation.get('Date')

    if not isinstance(dt_str, str):
        error = "The Field `Date` is not a string"
        return False, error

    try:
        datetime.fromisoformat(dt_str)
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

    categories_ok, error = check_categories(observation)
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

# End input validation functions
########################################

########################################
# Begin webserver


app = Flask(__name__)


@app.route('/should_search/', methods=['POST'])
def should_search():
    observation = request.get_json()

    observation_ok, error_message = check_observation(observation)
    if not observation_ok:
        return {"error": error_message}, 405

    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    predicted_outcome = True if proba > SUCCESS_RATE else False
    response = {'outcome': predicted_outcome}

    p = Prediction(
        observation=observation,
        proba=proba,
        predicted_outcome=predicted_outcome,
        observation_id=observation['observation_id'],
        type=observation['Type'],
        date=observation['Date'],
        part_of_a_policing_operation=observation['Part of a policing operation'],
        latitude=observation['Latitude'],
        longitude=observation['Longitude'],
        gender=observation['Gender'],
        age_range=observation['Age range'],
        officer_defined_ethnicity=observation['Officer-defined ethnicity'],
        legislation=observation['Legislation'],
        object_of_search=observation['Object of search'],
        station=observation['station'],
        hour=19,
        day_of_week='Friday',
        month=7
    )

    try:
        p.save()
        return response
    except IntegrityError:
        DB.rollback()
        error_msg = f"Observation Id: \'{observation['observation_id']}\' already exists"
        print(error_msg)
        error_response = {
            "error": error_msg
        }

        return error_response, 405


@app.route('/search_result/', methods=['POST'])
def search_result():
    observation = request.get_json()

    result_ok, error_message = check_result(observation)
    if not result_ok:
        return {"error": error_message}, 405

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
        return response
    except Prediction.DoesNotExist:
        error_msg = f"Observation Id: \'{observation['observation_id']}\' does not exist"
        error_response = {
            "error": error_msg
        }
        print(error_msg)
        return error_response, 405


@app.route('/list_data/')
def list_data():
    return [
        model_to_dict(obs) for obs in Prediction.select()
    ]

# End webserver
########################################


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
