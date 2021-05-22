from flask import Flask, request, render_template
APP = Flask(__name__, template_folder='templates')
from testChatBot import getpred
import traceback

@APP.route('/')
def random():
    return 'chatBot API'
    #return (render_template('main.html'))

@APP.route('/response', methods=['GET','POST'])
def getresponse():
    if request.method == 'GET':
        return (render_template('main.html'))

    if request.method == 'POST':
        sentence = request.form['string']
        try:
            resp = getpred(sentence)
            return (render_template('main.html', original_input=sentence, result=resp))

        except:

            return str(traceback.format_exc())
    #try:
        #sentence = request.args.get("string")
        #resp = getpred(sentence)
    #except:
        #print(traceback.format_exc())
        #resp = -1

if __name__ == '__main__':
    print('Initializing Keras chatBot')
    APP.run(host="0.0.0.0", port=4545, debug = True)

