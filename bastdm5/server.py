import os
import random
import string

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session)

import classification.prediction as cnn

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'bastdm5/uploads')
app = Flask(__name__)
app.secret_key = 'F12Zr47j\3yX R~X@H!jmM]Lwf/,?KT'


# Checks if file extension is .wav
def allowed_file_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in 'wav'


def generate_random_filename():
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    print "requesting " + path
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_sound_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect('/')
    file_to_upload = request.files['file']

    if file_to_upload and allowed_file_extension(file_to_upload.filename):
        filename = generate_random_filename() + '.wav'
        file_to_upload.save(os.path.join(UPLOAD_FOLDER, filename))
        return render_template('index.html', fname=filename)

    return render_template('index.html', errorMsg='There was an error uploading the file')


@app.route('/uploads/<filename>')
def get_uploaded_sound_file(filename):
    return send_from_directory('/app/bastdm5/uploads', filename)


@app.route('/predict', methods=['POST'])
def calculate_instrument_probabilities():
    file_to_predict = request.form['filename']
    prediction_probabilities, classified_instrument, score = cnn.predict_instrument(os.path.join(UPLOAD_FOLDER, file_to_predict))
    session['predictions'] = prediction_probabilities
    session['classified_instrument'] = classified_instrument
    session['score'] = round(score * 100, 2)
    return render_template('index.html', proba="True", fname=file_to_predict)


if __name__ == '__main__':
    cnn.initialize_model()
    app.run(host='0.0.0.0', port=9002, debug=False, threaded=True)
