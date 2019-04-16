from flask import Flask, url_for, render_template, redirect,request, flash
from werkzeug.utils import secure_filename
import os
from core.model import predict_image, read_img
import core.model as core
import cv2
import json
UPLOAD_FOLDER = 'D:\OneDrive\TFG\TFG_Python\static\images_received'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.secret_key = 'development'
app._static_folder = "./static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

folder = 'D:\OneDrive\TFG\TFG_Python\static\images_received'

@app.route('/test', methods=["POST"])
def test():
    isMobile=request.headers.get("isMobile")
    image = request.files.get("image")
    while image is None:
        pass
    filename = secure_filename(image.filename)
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    msg = "We have received that transmision, we send you back this message"
    print("He terminado")
    return json.dumps({"msg": msg, "TEST":"testing is done"})

@app.route('/')
def start():
    return render_template('start.html')

@app.route('/index')
def index():
    # TODO revisar esto en el momento en que haya multiples clientes accediendo a la app en el mismo tiempo
    clean_folder(folder)
    return render_template('index.html')

@app.route('/disclaimer', methods=['GET'])
def disclaimer_render():
    return render_template('disclaimer.html')

@app.route('/image/load')
def analize_render():
    errors = request.args.get('errors')
    return render_template('analysis.html', errors=errors)

@app.route('/image/analize', methods=['POST'])
def analize_image():
    if 'image' not in request.files:
        # TODO ponerle errores
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
        # TODO predict
        result = predict(filename, filepath)
        return redirect(url_for("image_result", filepath=filepath, filename=filename, result=result))
    else:
        errors = "Error: El fichero no tiene un nombre válido. Solo se aceptan imagenes en .png, .jpg, .jpeg y .gif"
        return redirect(url_for("analize_render", errors = errors))


def predict(filename, filepath, isMobile=False):
    result, core.model = predict_image(filepath, core.model)
    print(result)
    print(core.model)
    return result



@app.route('/image/result')
def image_result():
    filepath = request.args.get('filepath')
    filename = request.args.get('filename')
    result = request.args.get('result')

    reduced_img = read_img(filepath)
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

if __name__ == '__main__':
    app.run(debug=True)
