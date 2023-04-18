# @Author  : Edlison
# @Date    : 4/18/23 12:28
from flask import Flask, render_template, request
import json
from services import most, rating, inference


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/most', methods=['POST'])
def proxy_most():
    data = request.get_data()
    data = json.loads(data)
    limits = int(data['limits'])
    type = data['type']
    res = most(limits=limits, type=type)
    return {'data': res}


@app.route('/rating', methods=['POST'])
def proxy_rating():
    data = request.get_data()
    data = json.loads(data)
    limits = int(data['limits'])
    res = rating(limits=limits)
    return {'data': res}


@app.route('/infer', methods=['POST'])
def proxy_infer():
    data = request.get_data()
    data = json.loads(data)
    text = data['item_name']
    res = inference(text=text, from_openai=True)
    return {'data': res}


if __name__ == '__main__':
    app.run()

