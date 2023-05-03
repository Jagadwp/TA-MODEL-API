from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Bismillah. TA done on time Aamiin"
