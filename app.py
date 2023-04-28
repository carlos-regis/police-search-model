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
# Begin webserver

app = Flask(__name__)


@app.route('/should_search/', methods=['POST'])
def should_search():
    observation = request.get_json()
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
        error_msg = f"Observation Id: \'{observation['observation_id']}\' already exists"
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
        return response, 405


@app.route('/search_result/', methods=['POST'])
def search_result():
    obs_dict = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id ==
                           obs_dict['observation_id'])
        p.true_outcome = obs_dict['outcome']
        p.save()
        response = {
            "observation_id": p.observation_id,
            "outcome": p.true_outcome,
            "predicted_outcome": p.predicted_outcome
        }
        return response
    except Prediction.DoesNotExist:
        error_msg = f"Observation Id: \'{obs_dict['observation_id']}\' does not exist"
        response = {
            "error": error_msg
        }
        print(error_msg)
        return response, 405


@app.route('/list_data/')
def list_data():
    return [
        model_to_dict(obs) for obs in Prediction.select()
    ]

# End webserver
########################################


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
