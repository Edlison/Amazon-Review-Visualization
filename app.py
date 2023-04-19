# @Author  : Edlison
# @Date    : 4/18/23 12:28
from flask import Flask, render_template, request
import json
from services import most, rating, inference


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/graph')
def g():
    return render_template('graph.html')


@app.route('/graph_small')
def gs():
    return render_template('graph_small.html')


@app.route('/most', methods=['POST'])
def proxy_most():
    data = request.get_data()
    data = json.loads(data)
    limits = int(data['limits'])
    type = data['type']
    reverse = True if int(data['reverse']) == 1 else False
    res = most(limits=limits, type=type, reverse=reverse)
    return {'data': res}


@app.route('/rating', methods=['POST'])
def proxy_rating():
    data = request.get_data()
    data = json.loads(data)
    limits = int(data['limits'])
    reverse = True if int(data['reverse']) == 1 else False
    res = rating(limits=limits, reverse=reverse)
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

