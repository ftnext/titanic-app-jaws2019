from datetime import datetime

import boto3
from flask import (
    Flask,
    render_template,
    request,
    url_for,
)
from werkzeug import secure_filename


PREDICTION = {'0': '死亡', '1': '生存'}
AML_ENDPOINT = 'https://realtime.machinelearning.us-east-1.amazonaws.com'
ALLOWED_EXTENSIONS = set(['png', 'jpg'])
BUCKET_NAME = 'nikkie-jaws-2019-image-upload'
app = Flask(__name__)
client = boto3.client('machinelearning')
rek_client = boto3.client('rekognition')
s3 = boto3.client('s3')


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


def convert_input(input_data):
    """入力データをAmazon MLのAPIに送る形式に変換"""
    return {
        'Age': input_data['age'],
        'Pclass': input_data['pclass'],
        'Sex': input_data['sex'],
        'Embarked': input_data['embarked']
    }


def predict_by_amazonml(record):
    """recordをAmazon MLのAPIに送り、予測結果を取得"""
    response = client.predict(
        MLModelId='ml-jw1JEypL3C9',
        Record=record,
        PredictEndpoint=AML_ENDPOINT
    )
    return response['Prediction']['predictedLabel']


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form
    record = convert_input(input_data)
    predict_index = predict_by_amazonml(record)
    return render_template(
        'predict.html',
        prediction=PREDICTION[predict_index],
        input_data=input_data,
        img_url=None
    )


@app.route('/predict_by_image', methods=['POST'])
def predict_by_image():
    img_file = request.files['img_file']
    if img_file and allowed_file(img_file.filename):
        key = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-') \
         + secure_filename(img_file.filename)
        s3.upload_fileobj(img_file, BUCKET_NAME, key)
        params = {'Bucket': BUCKET_NAME, 'Key': key}
        img_url = s3.generate_presigned_url(
            'get_object',
            Params=params,
            ExpiresIn=300
        )
        response = rek_client.detect_faces(
            Image={
                'S3Object': {
                    'Bucket': BUCKET_NAME,
                    'Name': key
                }
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
        record = convert_input(input_data)
        predict_index = predict_by_amazonml(record)
        return render_template(
            'predict.html',
            prediction=PREDICTION[predict_index],
            input_data=input_data,
            img_url=img_url
        )
    else:
        return "Fileのアップロードに失敗しました"


if __name__ == "__main__":
    app.run(debug=True)
