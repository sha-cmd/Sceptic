import json
from flask import Flask, redirect, url_for, request
from src.content_based_filter import lambda_fct

app = Flask(__name__)


@app.route('/')
def hello_world():
    return """<form action="/userId" method="POST">
              <label for="fname">UserID:</label><br>
              <input type="text" id="userId" name="userId" value="1001"><br><br>
              <input type="submit" value="Submit">
              </form>"""


@app.route('/userId', methods=['POST'])
def greet():
    userId = request.form['userId']
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body":  lambda_fct(userId)
    }
