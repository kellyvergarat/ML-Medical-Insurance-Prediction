
import pickle
from flask import Flask
from flask import request
from flask import jsonify

input_file = 'ridge_model.bin'

with open(input_file, 'rb') as f_in: 
    model, test_data = pickle.load(f_in)

app = Flask('insurance_prediction')

@app.route('/predict', methods=['POST'])
def predict_price():
    customer = request.get_json()
    price_pred = float(model.predict(customer)[0])
    
    result = {
        'price_prediction': price_pred
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(port=9696, debug=True, host='0.0.0.0')