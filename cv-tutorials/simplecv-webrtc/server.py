from gevent import monkey; monkey.patch_all()
from flask import Flask, request, send_file

from socketio import socketio_manage
from socketio.namespace import BaseNamespace

from SimpleCV import *
from PIL import Image as PILImage
import os, re
import cStringIO as StringIO

HAAR_FEATURES = {
          'left_ear.xml': Color.GOLD
        , 'right_ear.xml': Color.GOLD
        , 'lefteye.xml': Color.GREEN
        , 'right_eye.xml': Color.GREEN
        , 'mouth.xml': Color.RED
        , 'nose.xml': Color.BLUE
 }

def find_haars(img):
    for haar_feature, color in HAAR_FEATURES.items():
        print haar_feature
        try:
            blob = img.findHaarFeatures(haar_feature)[0]
        except:
            blob = None
        print 'blob:', blob
        if blob:
            img.drawCircle((blob.x, blob.y), 30, color = color, thickness = 4)
    return img 

class SimpleCVNamespace(BaseNamespace):
    def on_image(self, data):
        print 'Image Received'
        imgstr = re.search(r'base64,(.*)', data).group(1) #This is a hack to clean up the encoding.
        tempimg = StringIO.StringIO(imgstr.decode('base64'))
        pilimg = PILImage.open(tempimg)
        img = Image(pilimg)
        #img = img.edges()
        img = find_haars(img)
        #pimg = img.getPIL()
        pimg = img
        output = StringIO.StringIO()
        #pimg.save(output, format='png')
        pimg.save(output)
        sendstr = 'data:image/jpeg;base64,' + output.getvalue().encode('base64')
        self.emit('update', sendstr)

# Flask routes
app = Flask(__name__)
@app.route('/')
def index():
    return send_file('static/index.html')

@app.route("/socket.io/<path:path>")
def run_socketio(path):
    socketio_manage(request.environ, {'': SimpleCVNamespace})

if __name__ == '__main__':
    print 'Listening on http://localhost:8080'
    app.debug = True
    from werkzeug.wsgi import SharedDataMiddleware
    app = SharedDataMiddleware(app, {
        '/': os.path.join(os.path.dirname(__file__), 'static')
        })
    from socketio.server import SocketIOServer
    SocketIOServer(('0.0.0.0', 8080), app,
        namespace="socket.io", policy_server=False).serve_forever()
    
