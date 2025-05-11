import joblib
import numpy as np
from config.path_config import MODEL_DIR
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

loaded_model = joblib.load(MODEL_DIR)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        # Get the input data from the form
        lead_time = int(request.form['lead_time'])
        no_of_special_requests = int(request.form['no_of_special_requests'])
        avg_price_per_room = float(request.form['avg_price_per_room'])
        arrival_date = int(request.form['arrival_date']) 
        arrival_month = int(request.form['arrival_month']) 
        no_of_week_nights = int(request.form['no_of_week_nights'])
        market_segment_type = int(request.form['market_segment_type']) 
        no_of_weekend_nights = int(request.form['no_of_weekend_nights'])
        no_of_adults = int(request.form['no_of_adults'])
        room_type_reserved = int(request.form['room_type_reserved'])
        
        features = np.array([[lead_time, no_of_special_requests, avg_price_per_room, arrival_date, arrival_month, no_of_week_nights, market_segment_type, no_of_weekend_nights, no_of_adults, room_type_reserved]])
        prediction = loaded_model.predict(features)
        
        print(f"Prediction: {prediction}")
        
        return render_template('index.html', prediction=prediction[0])
    
    else:
        return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
