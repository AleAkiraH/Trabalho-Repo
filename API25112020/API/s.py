from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/", methods=['GET'])
def getsimples():
	return "Api IGOR - {}".format(datetime.datetime.now())

if __name__ == '__main__':
	print ('API_IGOR')
	app.run(host='127.0.0.1', port=3000)