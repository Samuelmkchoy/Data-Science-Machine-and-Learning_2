from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model_path = '/Users/isabellaw/Desktop/Data Science Machine and Learning/Bayesian Classifiers/tech_politics_model.pkl'
model = joblib.load(model_path)

categories = ['comp.sys.mac.hardware', 'comp.windows.x', 'rec.autos', 'talk.politics.mideast']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json(force=True)
        text = data.get('text')
        if not text:
            return jsonify({'error': 'Missing text for prediction'}), 400
        
        prediction_index = model.predict([text])[0]
        
        predicted_category = categories[prediction_index]
        
        return jsonify({'predicted_category': predicted_category})
    except Exception as e:
    
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
     app.run(host='127.0.0.1', port=5003, debug=True)

