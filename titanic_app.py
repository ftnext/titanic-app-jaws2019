import boto3
from flask import (
    Flask,
    render_template,
    request,
    url_for,
)


PREDICTION = {'0': '死亡', '1': '生存'}
AML_ENDPOINT = 'https://realtime.machinelearning.us-east-1.amazonaws.com'
ALLOWED_EXTENSIONS = set(['png', 'jpg'])
app = Flask(__name__)
client = boto3.client('machinelearning')
rek_client = boto3.client('rekognition')


def allowed_file(filename):
    # 右の.で最大1回分割（拡張子を取り出す）
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/hello")
def hello():
    return "Hello World!"


@app.route('/')
def show_titanic_form():
    return render_template('show_titanic_form.html')


@app.route('/upload')
def show_image_upload_form():
    return render_template('show_image_upload_form.html')


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


@app.route('/predict_by_image', methods=['POST'])
def predict_by_image():
    img_file = request.files['img_file']
    if img_file and allowed_file(img_file.filename):
        response = rek_client.detect_faces(
            Image={
                # bytes-like に変更
                'Bytes': img_file.stream._file.read()
            },
            Attributes=['ALL']
        )
        detected_face = response['FaceDetails'][0]
        age_low = detected_face['AgeRange']['Low']
        age_high = detected_face['AgeRange']['High']
        age = (int(age_low) + int(age_high)) / 2
        sex = detected_face['Gender']['Value'].lower()
        input_data = {
            'age': str(age),
            'pclass': '3',
            'sex': sex,
            'embarked': 'S'
        }
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
    else:
        return "Fileのアップロードに失敗しました"


if __name__ == "__main__":
    app.run(debug=True)
