from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
import os
from vocal_analyzer import VocalAnalyzer
from generate_report import generate_report_context

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['audio_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        analyzer = VocalAnalyzer()
        analysis = analyzer.analyze_file(file_path)
        context = generate_report_context(analysis, "Uploaded Artist", 1)
        return render_template('report_template.html', analysis=analysis, filename=file.filename, report_context=context)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)