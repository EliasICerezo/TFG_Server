from flask import Flask, url_for, render_template, redirect,request, flash
from werkzeug.utils import secure_filename
import os
from core.modelfunctions import read_image, model_load_from_h5, predict_image
import core.model as core
import cv2
import json

UPLOAD_FOLDER = 'D:\OneDrive\TFG\TFG_Python\static\images_received'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])



app = Flask(__name__)


folder = 'D:\OneDrive\TFG\TFG_Python\static\images_received'

@app.route('/test', methods=["POST"])
def analyze_new_apps():
    image = request.files.get("image")
    while image is None:
        pass
    filename = secure_filename(image.filename)
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result = predict(filepath)
    print(result[0])
    dictionary ={"veredict": result[0]}
    clean_folder(app.config['UPLOAD_FOLDER'])
    return json.dumps(dictionary)

@app.route('/')
def start():
    clean_folder(UPLOAD_FOLDER)
    return render_template('start.html')


@app.route('/disclaimer', methods=['GET'])
def disclaimer_render():
    return render_template('disclaimer.html')

@app.route('/image/load')
def analize_render():
    errors = request.args.get('errors')
    clean_folder(UPLOAD_FOLDER)
    return render_template('analysis.html', errors=errors)

@app.route('/image/analize', methods=['POST'])
def analize_image():
    if 'image' not in request.files:
        errors = "Error: No se encuentra el fichero."
        return redirect(url_for("analize_render", errors=errors))
    file = request.files['image']
    if file.filename == '':
        errors = "Error: No has seleccionado ningun fichero de tu sistema."
        return redirect(url_for("analize_render", errors=errors))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        result = predict(filepath)
        return redirect(url_for("image_result", filepath=filepath, filename=filename, result=result))
    else:
        errors = "Error: El fichero no tiene un nombre valido. Solo se aceptan imagenes en .png, .jpg, .jpeg y .gif"
        return redirect(url_for("analize_render", errors =errors))


def predict(filepath):
    result, core.model = predict_image(filepath, core.model)
    return result

def load_model():
    core.model= model_load_from_h5(None)


@app.route('/image/result')
def image_result():
    filepath = request.args.get('filepath')
    filename = request.args.get('filename')
    result = request.args.get('result')

    reduced_img = read_image(filepath)
    newfilename, file_extension = os.path.splitext(filename)
    newfilename = newfilename+"2"+file_extension
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], newfilename), reduced_img)
    return render_template('result.html', filepath=filepath, filename=filename, newfilename=newfilename, result=result)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_folder(folderpath):

    for the_file in os.listdir(folderpath):
        file_path = os.path.join(folderpath, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
