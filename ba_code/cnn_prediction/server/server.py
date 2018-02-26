from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session)

import os
import random
import string
import ba_code.cnn_prediction.prediction as predict

ROOT_DIR = '/Users/yast/projects/20_hobby/music-instument-recognition/ba-code/ba_code/cnn_prediction/server/'
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'uploads')

app = Flask(__name__)
app.secret_key = 'F12Zr47j\3yX R~X@H!jmM]Lwf/,?KT'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Checks if file extension is .wav
def allowed_file_extension(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in 'wav'


# Generates a random 8 char string
def generate_filename():
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))


@app.route('/')
def index():
    return render_template('index.html')


# TODO: redirect everything else to index.html


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect('/')
    file_to_upload = request.files['file']

    if file_to_upload and allowed_file_extension(file_to_upload.filename):
        filename = generate_filename() + '.wav'
        file_to_upload.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        os.path.join(ROOT_DIR, filename)
        return render_template('index.html', fname=filename)

    return render_template('index.html', errorMsg="There was an error...sorry!")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/predict', methods=['POST'])
def predict_probabilities():
    filename = request.form['filename']
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    proba_predictions, classified_instrument, score = predict.predict_instrument(filepath)
    session['predictions'] = proba_predictions
    session['classified_instrument'] = classified_instrument
    session['score'] = round(score * 100, 2)
    return render_template('index.html', proba="True", fname=filename)


if __name__ == "__main__":
    predict.initialize_cnn()
    app.run(host='0.0.0.0', port=9002, debug=False, threaded=True)
