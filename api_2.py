from flask import Flask, request, jsonify
from api import lambda_handler
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/user', methods=['GET', 'POST'])
def user():
    if request.method == 'POST':
        if 'file' not in request.files:
            return {'error': 'No file part'}, 400
        file = request.files['file']
        predicted_normalised, durations = lambda_handler(file)
        # Fetch user logic here
        return jsonify({'notes': predicted_normalised, 'duration': durations})



app.run(host="0.0.0.0", port=5010)


