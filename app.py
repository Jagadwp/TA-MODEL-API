from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.pipeline import Pipeline

app = Flask(__name__)


@app.route("/get_result/hewani", methods=['POST'])
def get_result_hewani():
    # Read the CSV file
    data = pd.read_csv("./dataset/hewani/app_history.csv",
                       sep=';', on_bad_lines='skip')

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
    json_data = request.get_json()
    event_logs = json_data['event_logs']
    df = pd.DataFrame(event_logs)

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
        "status_halal": status_halal,
        "list_potensi": list_potensi
    }

    return jsonify(response), 200


if __name__ == "__main__":
    app.run()
