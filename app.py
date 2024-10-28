from flask import Flask, request, jsonify
import pandas as pd
from tsfresh import extract_features
from joblib import load

app = Flask(__name__)

# Load the pre-trained classifier model
model_path = "accelerometer_model_new.joblib"
classifier = load(model_path)

def add_sequence_id(df, sequence_id):
    df['sequence_id'] = sequence_id
    return df

def extract_tsfresh_features(ts_df):
    features = extract_features(ts_df, column_id="sequence_id", column_sort="time")
    return features

def preprocess_data(df):
    test_time_series = add_sequence_id(df, 1)
    test_features = extract_tsfresh_features(test_time_series)
    
    columns_to_add = [
        'classification__mean_n_absolute_max__number_of_maxima_7',
        'Ay__number_crossing_m__m_-1',
        'Ay__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.0',
        'SV_total__mean_n_absolute_max__number_of_maxima_7',
        'classification__ar_coefficient__coeff_1__k_10',
        'SV_total__ratio_beyond_r_sigma__r_0.5',
        'Az__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6',
        'SV_total__energy_ratio_by_chunks__num_segments_10__segment_focus_8',
        'Ax__autocorrelation__lag_5',
        'Az__ratio_beyond_r_sigma__r_1'
    ]

    missing_columns = [col for col in columns_to_add if col not in test_features.columns]
    for col in missing_columns:
        test_features[col] = 0

    selected_test_features = test_features[columns_to_add]
    return selected_test_features

@app.route('/predict', methods=['POST'])
def make_prediction():
    try:
        data = request.get_json()
        print("Received Data:", data)  # Log the incoming data
        
        if 'accelerometer_data' not in data:
            return jsonify({'error': 'Missing accelerometer data'}), 400
        
        # Log the accelerometer data
        print("Accelerometer Data:", data['accelerometer_data'])
        
        # Verify that the accelerometer data has the correct format (5 columns)
        if len(data['accelerometer_data'][0]) != 5:
            return jsonify({'error': 'Incorrect data format. Expected 5 columns.'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['accelerometer_data'], columns=['time', 'SV_total', 'Ax', 'Ay', 'Az'])
        
        # Continue processing...
        return jsonify({'success': 'Data processed successfully'})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
