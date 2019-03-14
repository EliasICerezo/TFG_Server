from flask import Flask, url_for, render_template, redirect,request, flash
from werkzeug.utils import secure_filename
import os
from core.model import predict_image
import core.model as core
UPLOAD_FOLDER = 'D:\OneDrive\TFG\TFG_Python\images_received'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.secret_key = 'development'
app._static_folder = "./static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/')
def start():
    return render_template('start.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/disclaimer', methods=['GET'])
def disclaimer_render():
    return render_template('disclaimer.html')

@app.route('/image/load')
def analize_render():
    return render_template('analysis.html')

@app.route('/image/analize', methods=['POST'])
def analize_image():
    if 'image' not in request.files:
        # TODO ponerle errores
        return redirect(url_for("analize_render"))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for("analize_render"))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        # TODO predict
        result, core.model = predict_image(filepath,core.model)
        print(result)
        print(core.model)
        return redirect(url_for("image_result",filename=filename))
    else:
        return redirect(url_for("analize_render"))

@app.route('/image/result')
def image_result():

    return redirect(url_for('index'))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=True)
