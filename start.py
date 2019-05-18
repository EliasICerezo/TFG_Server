from tornado.wsgi import WSGIContainer
from tornado.ioloop import IOLoop
from tornado.web import FallbackHandler, RequestHandler, Application
from views import app
from flask_cors import CORS

UPLOAD_FOLDER = 'D:\OneDrive\TFG\TFG_Python\static\images_received'

class MainHandler(RequestHandler):
    def get(self):
        self.write("This message comes from Tornado ^_^")


app.secret_key = 'development'
app._static_folder = "./static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)
tr = WSGIContainer(app)

application = Application([
    (r"/tornado", MainHandler),
    (r".*", FallbackHandler, dict(fallback=tr)),
])

if __name__ == "__main__":
    application.listen(80, address='0.0.0.0')
    IOLoop.instance().start()
