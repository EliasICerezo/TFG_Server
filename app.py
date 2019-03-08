from flask import Flask, url_for, render_template


app = Flask(__name__)
app.debug = True
app.secret_key = 'development'
app._static_folder = "./static"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/disclaimer')
def disclaimer_render():
    return render_template('disclaimer.html')

@app.route('/image/analize')
def analize_image():
    pass

@app.route('/image/result')

@app.route('/about')
def about_render():
    pass

if __name__ == '__main__':
    app.run()
