from flask import Flask
app = Flask(__name__)
@app.route("/members")
def members():
    #return {"members":["Member1","Member2","Member3"]}
    try:
        with open('../emo.txt', 'r') as file:
            return file.read()
    except FileNotFoundError:
        return 'File not found', 404
if __name__=="__main__":
    app.run(debug=True)
