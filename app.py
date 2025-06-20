from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# Load ML model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fname = db.Column(db.String(100))
    lname = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

# Create DB file
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return render_template('home.html')

    
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fname = request.form['fname']
        lname = request.form['lname']
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']

        if password != confirmpassword:
            return render_template('register.html', error="Passwords do not match")

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template('register.html', error="Email already registered")

        hashed_password = generate_password_hash(password)
        new_user = User(fname=fname, lname=lname, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        session['user'] = email
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user'] = user.email
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid email or password")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return render_template('logout.html')

@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        fields = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
                  'CHRONIC_DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING',
                  'COUGHING', 'SHORTNESS_OF_BREATH', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN']
        features = []

        for field in fields:
            val = request.form.get(field)
            if val is None or val.strip() == '':
                raise ValueError(f"Missing value for {field}")
            features.append(int(val))

        prediction = model.predict([features])[0]

        # Check if model supports probability
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba([features])[0][1]  # Probability of class 1 (lung cancer)
            cancer_prob_percent = probability * 100
            result = 'Positive for Lung Cancer' if prediction == 1 else 'Negative for Lung Cancer'
            return render_template('index.html', prediction_text=f'Result: {result} ({cancer_prob_percent:.2f}% chance of lung cancer)')
        else:
            result = 'Positive for Lung Cancer' if prediction == 1 else 'Negative for Lung Cancer'
            return render_template('index.html', prediction_text=f'Result: {result} (Probability not available)')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
