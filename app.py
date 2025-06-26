from dotenv import load_dotenv
load_dotenv()
# app.py
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Import utility functions
from utils import get_granite_response, get_sentiment

app = Flask(__name__)
app.secret_key = os.urandom(24)

# In-memory user and chat data (can be replaced with DB)
users = {"citizen1": {"password": "password123"}}
chat_history = {}
feedback_data = []

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            login_user(User(username))
            flash('Logged in successfully.', 'success')
            return redirect(url_for('chat'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    user_id = current_user.id
    if user_id not in chat_history:
        chat_history[user_id] = []

    if request.method == 'POST':
        user_message = request.form['message']
        chat_history[user_id].append({'sender': 'user', 'text': user_message})

        context_prompt = f"""
        You are 'Citizen AI', a helpful and respectful assistant for a government citizen engagement platform.
        A citizen is asking a question. Provide a clear, concise, and helpful response.
        Citizen's question: "{user_message}"
        Your response:
        """
        ai_response = get_granite_response(context_prompt)
        chat_history[user_id].append({'sender': 'ai', 'text': ai_response})

        sentiment = get_sentiment(user_message)
        feedback_data.append({
            'username': user_id,
            'text': user_message,
            'sentiment': sentiment['label'],
            'sentiment_score': sentiment['score']
        })

    return render_template('chat.html', history=chat_history[user_id])

@app.route('/dashboard')
@login_required
def dashboard():
    if not feedback_data:
        return render_template('dashboard.html', total_feedback=0, sentiment_chart=None, recent_concerns=[])

    df = pd.DataFrame(feedback_data)
    total_feedback = len(df)
    sentiment_counts = df['sentiment'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, hole=.3)])
    fig.update_layout(title_text='Citizen Sentiment Distribution')
    sentiment_chart_html = pio.to_html(fig, full_html=False)
    recent_concerns = df[df['sentiment'] == 'negative'].tail(5).to_dict('records')

    return render_template('dashboard.html',
        total_feedback=total_feedback,
        sentiment_chart=sentiment_chart_html,
        recent_concerns=recent_concerns
    )

if __name__ == '__main__':
    app.run(debug=True)
