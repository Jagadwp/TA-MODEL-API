from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.pipeline import Pipeline
from config import Config
from flask_cors import CORS

app = Flask(__name__)
app.config.from_object(Config)

# Enable CORS
cors = CORS(app, resources={
    r"/*": {"origins": ["http://127.0.0.1:8000", "https://siapsertifikasihalal.my.id"]}
}, supports_credentials=True)


@app.route("/", methods=['GET'])
def home():
    return "TA MODEL API -> Run in port:" + str(Config.PORT)


@app.route("/prediction/hewani", methods=['POST'])
def get_prediction_result_hewani():
    # Read the CSV file
    data = pd.read_csv(Config.HEWANI_DATASET_PATH,
                       sep=';', on_bad_lines='skip')

    response = predict(data, request)
    return response, 200


@app.route("/prediction/nabati", methods=['POST'])
def get_prediction_result_nabati():
    # Read the CSV file
    data = pd.read_csv(Config.NABATI_DATASET_PATH,
                       sep=';', on_bad_lines='skip')

    response = predict(data, request)
    return response, 200


def predict(data, request):
    # Pivot the data table
    dataset = data.pivot_table(
        index="CaseID", columns="Activity", values="Status_Halal", aggfunc='first')

    # Drop rows with missing values
    dataset = dataset.dropna(subset=['Ambil Kesimpulan'])

    # Separate features and labels
    feature = dataset.drop("Ambil Kesimpulan", axis=1)
    label = dataset["Ambil Kesimpulan"]

    # Encode categorical features
    enc = OneHotEncoder(handle_unknown='ignore')
    feature_enc = enc.fit_transform(feature)

    # Define parameter grid for GridSearchCV
    param_grid = {'C': [10], 'gamma': [1, 0.1, 0.01, 0.001],
                  'kernel': ['poly', 'rbf', 'sigmoid']}

    # Perform GridSearchCV with SVM
    grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
    grid.fit(feature_enc, label)

    # Process request data
    df = arrangeData(request)

    # Preprocess test data
    df_test = df.pivot_table(
        index="CaseID", columns="Activity", values="Status_Halal", aggfunc='first')
    test_data = pd.concat([pd.DataFrame(columns=dataset.columns), df_test])
    test_data = test_data.drop("Ambil Kesimpulan", axis=1)

    # Get final result
    pipeline_svm = Pipeline([('enc', enc), ('grid', grid)])
    hasil_svm = pipeline_svm.predict(test_data)
    status_halal = hasil_svm[0]

    # Get activities with 'Haram' status
    potensi = df.loc[df['Status_Halal'] == 'Haram', 'Activity']
    list_potensi = potensi.tolist()

    # Create response dictionary
    response = {
        "status-halal": status_halal,
        "list-potensi": list_potensi
    }

    return jsonify(response)


def arrangeData(request):
    json_data = request.get_json()
    event_logs = json_data['event-log']

    df_data = []
    for log in event_logs:
        case_id = json_data['ingredient-id']
        activity = log['label']
        timestamp = log['timestamp']
        originator = json_data['user-id']
        status_halal = log['value']

        df_data.append({
            'CaseID': case_id,
            'Activity': activity,
            'Timestamp': timestamp,
            'Originator': originator,
            'Status_Halal': status_halal
        })

    return pd.DataFrame(df_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Config.PORT)
