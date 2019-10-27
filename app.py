from q_learning import initialize_q_learning, step_q_learning

from flask import Flask, jsonify, request, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize the q tables
@app.route('/api/v1.0/initialize', methods=['POST'])
def initialize():
    if not request.json or not 'time' in request.json or not 'price' in request.json:
        abort(404)
    return jsonify(initialize_q_learning(request.json['price'],request.json['time']))

# Step the q tables
@app.route('/api/v1.0/step', methods=['POST'])
def step():
    # if not request.json or not 'q_table_e' in request.json or not 'q_table_r' in  \
    #     request.json or not 'last_s' in request.json or not 'last_e' in request.json \
    #          or not 'last_r' in request.json or not 'satisfaction' in resuest.json \
    #             or not 'episode' in request.json:
    #     abort(404)
    return jsonify(step_q_learning(request.json['q_table_e'], request.json['q_table_r'], \
         request.json['last_s'], request.json['last_e'], request.json['last_r'], \
            request.json['satisfaction'], request.json['episode'], request.json['price'], \
                request.json['time']))

if __name__ == '__main__':
    app.run()