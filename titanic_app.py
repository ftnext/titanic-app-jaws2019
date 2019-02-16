import boto3
from flask import (
    Flask,
    render_template,
    request,
    url_for,
)


PREDICTION = {'0': '死亡', '1': '生存'}
AML_ENDPOINT = 'https://realtime.machinelearning.us-east-1.amazonaws.com'
app = Flask(__name__)
client = boto3.client('machinelearning')


@app.route("/hello")
def hello():
    return "Hello World!"


@app.route('/')
def show_titanic_form():
    return render_template('show_titanic_form.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form
    record = {
        'Age': input_data['age'],
        'Pclass': input_data['pclass'],
        'Sex': input_data['sex'],
        'Embarked': input_data['embarked']
    }
    response = client.predict(
        MLModelId='ml-jw1JEypL3C9',
        Record=record,
        PredictEndpoint=AML_ENDPOINT
    )
    predict_index = response['Prediction']['predictedLabel']
    return render_template(
        'predict.html',
        prediction=PREDICTION[predict_index],
        input_data=input_data
    )


if __name__ == "__main__":
    app.run(debug=True)
