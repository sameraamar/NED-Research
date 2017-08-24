from flask import Flask,request
from flaskext.mysql import MySQL

mysql = MySQL()
app = Flask(__name__)
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'mysql'
app.config['MYSQL_DATABASE_DB'] = 'thesis_2017'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)


@app.route("/")
def hello():
    return "Welcome to Python Flask App!"

@app.route("/Authenticate")
def Authenticate():
    tweetId = request.args.get('tweetid')
    cursor = mysql.connect().cursor()
    cursor.execute("SELECT * from dataset where tweet_id=" + tweetId + "")
    data = cursor.fetchone()
    if data is None:
        return "Username or Password is wrong"
    else:
        return "Logged in successfully"

@app.route('/jinjaman')
def jinjaman():
	try:
		data = [15, '15', 'Python is good','Python, Java, php, SQL, C++','<p><strong>Hey there!</strong></p>']
		return render_template("index.html", data=data)
	except Exception as e:
		return(str(e))

if __name__ == "__main__":
    app.run()