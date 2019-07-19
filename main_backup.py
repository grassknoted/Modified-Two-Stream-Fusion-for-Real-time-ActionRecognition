from flask import Flask, render_template, Response
from handwash_system import HandwashSystem
import signal
import sys

step_image = [
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done",
  "not_done"]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', step_image=step_image)

def gen(live_stream):
    while True:
        frame = live_stream.get_frame()[0]
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(HandwashSystem()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def handler(signal, frame):
  print('CTRL-C pressed!')
  sys.exit(0)

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', debug=True, port = 81)