from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
import os
from vocal_analyzer import VocalAnalyzer
from generate_report import generate_report_context
from werkzeug.utils import secure_filename
import numpy as np
from datetime import datetime
import tempfile
import shutil

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Replace with a proper secret key in production

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def aggregate_analyses(analyses):
    """Aggregate multiple analyses into a single comprehensive report."""
    if not analyses:
        return None

    # Initialize aggregate data
    aggregate = analyses[0].copy()

    # Calculate averages for numerical values
    for key in aggregate:
        if isinstance(aggregate[key], (int, float, np.integer, np.floating)):
            values = [float(analysis[key]) for analysis in analyses]
            aggregate[key] = float(np.mean(values))
        elif isinstance(aggregate[key], (list, np.ndarray)):
            # For lists (like spectral_contrast), calculate element-wise mean
            values = np.array([analysis[key] for analysis in analyses])
            aggregate[key] = np.mean(values, axis=0).tolist()
        elif isinstance(aggregate[key], dict):
            # Handle nested dictionaries (like register_transitions)
            for subkey in aggregate[key]:
                if isinstance(aggregate[key][subkey], (int, float, np.integer, np.floating)):
                    aggregate[key][subkey] = float(aggregate[key][subkey])

    # Add multi-song specific metrics
    aggregate['number_of_songs_analyzed'] = len(analyses)
    aggregate['consistency_score'] = float(calculate_consistency(analyses))

    return aggregate

def calculate_consistency(analyses):
    """Calculate consistency across multiple performances."""
    if len(analyses) < 2:
        return 10.0  # Perfect score for single song

    # Calculate variance of key metrics
    key_metrics = ['pitch_accuracy', 'breath_control', 'resonance_score', 'articulation_score']
    variances = []

    for metric in key_metrics:
        values = [float(analysis[metric]) for analysis in analyses]
        variance = float(np.var(values))
        variances.append(variance)

    # Convert variance to consistency score (lower variance = higher consistency)
    avg_variance = float(np.mean(variances))
    consistency = float(10 * np.exp(-avg_variance))  # Scale to 0-10
    return min(10.0, consistency)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if artist name was submitted
        artist_name = request.form.get('artist_name', '').strip()
        if not artist_name:
            flash('Please enter an artist name', 'error')
            return redirect(request.url)

        # Check if any file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        files = request.files.getlist('file')
        if not files or all(file.filename == '' for file in files):
            flash('No files selected', 'error')
            return redirect(request.url)

        # Process each uploaded file
        temp_dir = tempfile.mkdtemp()
        analysis_results = None

        try:
            for file in files:
                if file and file.filename != '' and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(temp_dir, filename)
                    file.save(filepath)

                    # Analyze the audio file
                    analyzer = VocalAnalyzer()
                    try:
                        current_analysis = analyzer.analyze_file(filepath)
                        if current_analysis:
                            if analysis_results is None:
                                analysis_results = current_analysis
                            else:
                                # Merge analyses (for multiple files)
                                for key, value in current_analysis.items():
                                    if key in analysis_results and isinstance(value, (int, float)):
                                        analysis_results[key] = (analysis_results[key] + value) / 2
                    except Exception as e:
                        flash(f'Error analyzing {filename}: {str(e)}', 'error')
                        continue
                else:
                    flash(f'Invalid file type. Allowed types are: {", ".join(ALLOWED_EXTENSIONS)}', 'error')

            # Generate report if analysis was successful
            if analysis_results:
                context = generate_report_context(analysis_results, artist_name)
                if context:
                    return render_template('report_template.html', analysis=context)
                else:
                    flash('Error generating report', 'error')
            else:
                flash('No valid analysis results', 'error')

        except Exception as e:
            flash(f'Error processing files: {str(e)}', 'error')
        finally:
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)

        return redirect(request.url)

    # GET request - show upload form
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/test_report')
def test_report():
    """Generate a test report with sample data for testing the template."""
    sample_data = {
        'pitch_accuracy': 8.5,
        'breath_control': 7.2,
        'vibrato_quality': 8.1,
        'vibrato_rate': 5.5,
        'resonance': 7.8,
        'dynamic_range': 6.9,
        'articulation': 8.7,
        'lowest_note': 'C3',
        'highest_note': 'C5',
        'range_span': '2.0 octaves',
        'chest_to_mix': 'D4',
        'mix_to_head': 'E5',
        'head_to_whistle': 'C6',
        'consistency': 8.5
    }

    context = generate_report_context(sample_data, "Test Artist")
    return render_template('report_template.html', analysis=context)

if __name__ == '__main__':
    app.run(debug=True)