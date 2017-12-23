# Demo 1: Real-Time Web Demo
See [our YouTube video](https://www.youtube.com/watch?v=LZJOTRkjZA4)
of using this in a real-time web application
for face recognition.
The source is available in [demos/web](https://github.com/cmusatyalab/openface/blob/master/demos/web).


<a href='https://www.youtube.com/watch?v=LZJOTRkjZA4'><img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/youtube-web.gif'></img></a>

This demo does the full face recognition pipeline on every frame.
In practice, object tracking
[like dlib's](http://blog.dlib.net/2015/02/dlib-1813-released.html)
should be used once the face recognizer has predicted a face.

To run on your system, after following the setup directions
below, install the requirements for the web demo with
`./install-deps.sh` and `sudo pip install -r requirements.txt`
from the `demos/web` directory.
This is currently not included in the Docker container.
The application is split into a processing server and static
web pages that communicate via web sockets.

Start the server with `./demos/web/server.py`.
With your client system with webcam and browser,
you should now be able to send a request to the websocket
connection with `curl your-server:9000` (`localhost:9000` if running on your machine),
which should inform you that it's' a WebSocket endpoint and not a web server.
Please check routing between your client and server if you
get connection refused issues.

If you are running the server remotely (relative to your browser)
or in a Docker container,
change the WebSocket connection in
[index.html](https://github.com/cmusatyalab/openface/blob/master/demos/web/index.html)
from `127.0.0.1` to the IP address of your server
that you were able to connect to with `curl`.
With the WebSocket server running, serve the static website with
`python2 -m SimpleHTTPServer 8000` from the `/demos/web` directory.
You should now be able to access the demo from your browser
at `http://your-server:8000`, (`http://localhost:8000` if running on your machine),
The saved faces are only available for the browser session.
