from flask import Flask, url_for, render_template, redirect


app = Flask(__name__)
app.secret_key = 'development'
app._static_folder = "./static"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/disclaimer', methods=['GET'])
def disclaimer_render():
    return render_template('disclaimer.html')

@app.route('/image/analize')
def analize_image():
    return render_template('analysis.html')

@app.route('/image/result')
def image_result():
    return (url_for('index'))

@app.route('/about')
def about_render():
    pass

if __name__ == '__main__':
    app.run(debug=True)
