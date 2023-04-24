import os
import json

import pandas as pd

import pickle
import joblib

from flask import Flask, request, jsonify
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

########################################
# Begin database

# DB = SqliteDatabase('predictions.db')
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')


class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    predicted_outcome = FloatField()
    true_outcome = TextField(null=True)

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


@app.route('/should_search', methods=['POST'])
def should_search():
    obs_dict = request.get_json()
    _id = obs_dict['observation_id']
    observation = obs_dict
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    predicted_outcome = pipeline.predict_proba(obs)[0, 1]
    response = {'outcome': predicted_outcome}

    p = Prediction(
        observation_id=_id,
        observation=observation,
        predicted_outcome=predicted_outcome
    )

    try:
        p.save()
    except IntegrityError:
        error_msg = f"Observation Id: '{_id}' already exists"
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()

    return jsonify(response)


@app.route('/search_result', methods=['POST'])
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
        return jsonify(response)
    except Prediction.DoesNotExist:
        error_msg = f"Observation Id: \'{obs_dict['observation_id']}\' does not exist"
        return jsonify({'error': error_msg})


@app.route('/list_data')
def list_data():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])

# End webserver
########################################


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
