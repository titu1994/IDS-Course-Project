import sys
sys.path.insert(0, "..")

import flask
from flask import request

from staging.ml_eval import predict_ml_ratings
from staging.dl_eval import predict_dl_ratings

# initialize the server
app = flask.Flask(__name__)

# initialize the models completely
print("Please wait while the models are being initialized for fast inference ! \n"
      "This may take several minutes...")

predict_ml_ratings._initialize()
predict_dl_ratings._initialize()

print("Server is ready !")


@app.route('/')
def default():
    return "Rating Server Initialized"


@app.route('/rating/ml', methods=['POST'])
def predict_rating_ml():
    if request.json is None:
        return flask.abort(400)

    if 'model_name' not in request.json:
        return flask.Response('"model_name" not provided in request !')

    if 'query' not in request.json:
        return flask.Response('"query" string not provided in request !')

    ml_model = str(request.json['model_name']).lower()
    query = request.json['query']

    return _resolve_ml_query(ml_model, query)


@app.route('/rating/dl', methods=['POST'])
def predict_rating_dl():
    if request.json is None:
        return flask.abort(400)

    if 'model_name' not in request.json:
        return flask.Response('"model_name" not provided in request !')

    if 'query' not in request.json:
        return flask.Response('"query" string not provided in request !')

    ml_model = str(request.json['model_name']).lower()
    query = request.json['query']

    return _resolve_dl_query(ml_model, query)


def _resolve_ml_query(model_name, query):
    if model_name not in ['tree', 'logistic', 'forest']:
        return flask.Response('Incorrect ml model name. Must be one of ["tree", "logistic", "forest"')

    if model_name == 'tree':
        rating, confidence = predict_ml_ratings.get_decision_tree_rating_prediction(query)
    elif model_name == 'logistic':
        rating, confidence = predict_ml_ratings.get_logistic_regression_rating_prediction(query)
    else:
        rating, confidence = predict_ml_ratings.get_random_forest_rating_prediction(query)

    response = {
        'rating': int(rating[0]),
        'confidence': confidence[0],
    }

    return flask.jsonify(**response)


def _resolve_dl_query(model_name, query):
    if model_name not in ['lstm', 'mult_lstm', 'malstm_fcn']:
        return flask.Response('Incorrect dl model name. Must be one of ["lstm", "mult_lstm", "malstm_fcn]"')

    if model_name == 'lstm':
        rating, confidence = predict_dl_ratings.get_lstm_ratings_prediction(query)
    elif model_name == 'mult_lstm':
        rating, confidence = predict_dl_ratings.get_multiplicative_lstm_ratings_prediction(query)
    else:
        rating, confidence = predict_dl_ratings.get_malstm_fcn_ratings_prediction(query)

    response = {
        'rating': int(rating[0]),
        'confidence': float(confidence[0]),
    }

    return flask.jsonify(**response)



if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    app.run(host='localhost', port=9001)