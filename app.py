from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash
from botvoi import get_voice_input, generate_interview_report, llm, graph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import tempfile

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///interviews.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    interviews = db.relationship('Interview', backref='user', lazy=True)

class Interview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    report = db.Column(db.Text, nullable=True)
    audio_path = db.Column(db.String(200), nullable=True)
    chat_history = db.Column(db.Text, nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        
        return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already exists')
        
        user = User(
            username=username,
            password=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    interviews = Interview.query.filter_by(user_id=current_user.id).order_by(Interview.date.desc()).all()
    return render_template('dashboard.html', interviews=interviews)

@app.route('/interview')
@login_required
def interview():
    return render_template('interview.html')

@app.route('/api/start-interview', methods=['POST'])
@login_required
def start_interview():
    # Create new interview record with initial chat history
    chat_history = [
        SystemMessage(content="You are an AI interview coach assistant. Conduct a professional mock interview, asking one question at a time. Wait for the user's response before continuing.")
    ]
    
    interview = Interview(
        user_id=current_user.id,
        chat_history=json.dumps([msg.dict() for msg in chat_history])
    )
    db.session.add(interview)
    db.session.commit()
    
    return jsonify({'interview_id': interview.id})

@app.route('/api/process-audio', methods=['POST'])
@login_required
def process_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    interview_id = request.form.get('interview_id')
    interview = db.session.get(Interview, interview_id)
    if not interview:
        return jsonify({'error': 'Interview not found'}), 404
    
    if interview.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Save audio file
    audio_file = request.files['audio']
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f'interview_{interview_id}.wav')
    audio_file.save(audio_path)
    
    # Process audio using botvoi functions
    transcription = get_voice_input(audio_path)
    
    # Get chat history and properly deserialize messages
    chat_history = []
    for msg in json.loads(interview.chat_history):
        if msg.get('type') == 'system':
            chat_history.append(SystemMessage(content=msg['content']))
        elif msg.get('type') == 'human':
            chat_history.append(HumanMessage(content=msg['content']))
        elif msg.get('type') == 'ai':
            chat_history.append(AIMessage(content=msg['content']))
    
    chat_history.append(HumanMessage(content=transcription))
    
    # Get AI response
    state = {"messgaes": chat_history}
    result = graph.invoke(state)
    chat_history = result["messgaes"]
    ai_response = chat_history[-1].content
    
    # Update interview record with properly serialized messages
    serialized_history = []
    for msg in chat_history:
        if isinstance(msg, SystemMessage):
            serialized_history.append({'type': 'system', 'content': msg.content})
        elif isinstance(msg, HumanMessage):
            serialized_history.append({'type': 'human', 'content': msg.content})
        elif isinstance(msg, AIMessage):
            serialized_history.append({'type': 'ai', 'content': msg.content})
    
    interview.chat_history = json.dumps(serialized_history)
    interview.audio_path = audio_path
    db.session.commit()
    
    # Generate analysis
    analysis = {
        'tone': 0.8,  # Placeholder - implement actual tone analysis
        'grammar_errors': 2,  # Placeholder - implement actual grammar analysis
        'relevance': 0.9  # Placeholder - implement actual relevance analysis
    }
    
    return jsonify({
        'transcription': transcription,
        'response': ai_response,
        'analysis': analysis
    })

@app.route('/api/end-interview/<int:interview_id>', methods=['POST'])
@login_required
def end_interview(interview_id):
    interview = db.session.get(Interview, interview_id)
    if not interview:
        return jsonify({'error': 'Interview not found'}), 404
    
    if interview.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Get chat history and properly deserialize messages
        chat_history = []
        for msg in json.loads(interview.chat_history):
            if msg.get('type') == 'system':
                chat_history.append(SystemMessage(content=msg['content']))
            elif msg.get('type') == 'human':
                chat_history.append(HumanMessage(content=msg['content']))
            elif msg.get('type') == 'ai':
                chat_history.append(AIMessage(content=msg['content']))
        
        # Generate report
        report = generate_interview_report(interview.audio_path, chat_history)
        
        if report is None:
            app.logger.error("Failed to generate report - report is None")
            return jsonify({'error': 'Failed to generate report'}), 500
            
        # Save report to database
        interview.report = json.dumps(report)
        db.session.commit()
        
        return jsonify({'success': True, 'report_id': interview.id})
    except Exception as e:
        app.logger.error(f"Error in end_interview: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/report/<int:interview_id>')
@login_required
def view_report(interview_id):
    interview = Interview.query.get_or_404(interview_id)
    if interview.user_id != current_user.id:
        return redirect(url_for('dashboard'))
    report = json.loads(interview.report) if interview.report else None
    return render_template('report.html', interview=interview, report=report)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 