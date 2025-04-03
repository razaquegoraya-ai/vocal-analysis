from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash, jsonify
import os
from vocal_analyzer import VocalAnalyzer
from generate_report import generate_report_context, calculate_trends, generate_comparative_insights, generate_overall_recommendations
from werkzeug.utils import secure_filename
import numpy as np
from datetime import datetime
import tempfile
import shutil
import logging
from logging.handlers import RotatingFileHandler
import gc
import traceback
import psutil
import time

app = Flask(__name__)

# Configure logging with more detailed formatting
if not os.path.exists('logs'):
    os.makedirs('logs')
file_handler = RotatingFileHandler('logs/app.log', maxBytes=1024*1024, backupCount=10)  # Increased log size
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]\n'
    'Memory Usage: %(memory_usage).2f MB'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Add memory usage to log record
old_factory = logging.getLogRecordFactory()
def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    record.memory_usage = get_memory_usage()
    return record
logging.setLogRecordFactory(record_factory)

app.logger.info('Vocal Analysis startup')

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def aggregate_analyses(analyses):
    """Aggregate multiple analyses into a single result"""
    if not analyses:
        return None
    
    app.logger.info(f'Starting aggregation of {len(analyses)} analyses')
    
    # Define all possible metrics with their mappings
    metrics_mapping = {
        'pitch_accuracy': 'pitch_accuracy',
        'breath_control': 'breath_control',
        'resonance': 'resonance',
        'dynamic_range': 'dynamic_range',
        'vibrato_rate': 'vibrato_rate',
        'vibrato_extent': 'vibrato_extent'
    }
    
    aggregate = {}
    
    # Aggregate each metric
    for output_key, input_key in metrics_mapping.items():
        try:
            # Get values for this metric from all analyses that have it
            values = []
            for analysis in analyses:
                if input_key in analysis:
                    value = analysis[input_key]
                    if isinstance(value, (int, float)):
                        values.append(float(value))
                    app.logger.info(f'Found value for {input_key}: {value}')
            
            if values:  # Only calculate if we have values
                aggregate[output_key] = sum(values) / len(values)
                app.logger.info(f'Aggregated {output_key}: {aggregate[output_key]}')
            else:
                app.logger.warning(f'No values found for metric: {input_key}')
                aggregate[output_key] = 0.0  # Default value if metric is missing
        except Exception as e:
            app.logger.error(f'Error aggregating metric {input_key}: {str(e)}')
            aggregate[output_key] = 0.0  # Default value on error
    
    # Copy non-numeric values from the first analysis that has them
    for key in ['lowest_note', 'highest_note', 'range_span', 'chest_to_mix', 'mix_to_head', 'head_to_whistle']:
        for analysis in analyses:
            if key in analysis and analysis[key]:
                aggregate[key] = analysis[key]
                app.logger.info(f'Using {key} from analysis: {aggregate[key]}')
                break
    
    app.logger.info(f'Final aggregated result: {aggregate}')
    return aggregate

def calculate_consistency(analyses):
    """Calculate consistency across multiple recordings"""
    if not analyses:
        return 0.0
    
    # Define metrics to check for consistency
    metrics = [
        'pitch_accuracy', 'breath_control', 'resonance',
        'dynamic_range', 'vibrato_rate', 'vibrato_extent'
    ]
    
    consistency_scores = []
    
    for metric in metrics:
        try:
            # Get values for this metric from all analyses that have it
            values = [float(analysis[metric]) for analysis in analyses if metric in analysis]
            if len(values) >= 2:  # Need at least 2 values to calculate consistency
                # Calculate standard deviation
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                std_dev = variance ** 0.5
                
                # Convert to consistency score (inverse of standard deviation)
                # Higher std_dev means lower consistency
                consistency = max(0, 10 - std_dev)
                consistency_scores.append(consistency)
        except Exception as e:
            app.logger.error(f'Error calculating consistency for {metric}: {str(e)}')
    
    # Return average consistency if we have scores, otherwise 0
    return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0

def process_audio_file(filepath, analyzer):
    """Process a single audio file with memory optimization"""
    start_time = time.time()
    app.logger.info(f'Starting analysis of {os.path.basename(filepath)}')
    
    try:
        # Clear memory before processing
        gc.collect()
        
        # Process the file
        current_analysis = analyzer.analyze_file(filepath)
        
        # Log the analysis results
        app.logger.info(f'Analysis results: {current_analysis}')
        
        # Log processing time and memory usage
        processing_time = time.time() - start_time
        app.logger.info(f'Completed analysis of {os.path.basename(filepath)} in {processing_time:.2f} seconds')
        
        return current_analysis
    except Exception as e:
        app.logger.error(f'Error analyzing {os.path.basename(filepath)}: {str(e)}\n{traceback.format_exc()}')
        return None
    finally:
        # Force garbage collection after processing
        gc.collect()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            app.logger.info('Processing upload request')
            if 'audio_file1' not in request.files and 'audio_file2' not in request.files and 'audio_file3' not in request.files:
                app.logger.error('No file part')
                return 'No file part', 400
            
            uploaded_files = []
            temp_dir = tempfile.mkdtemp()
            
            # Save files first
            for file_key in ['audio_file1', 'audio_file2', 'audio_file3']:
                if file_key in request.files:
                    file = request.files[file_key]
                    if file.filename:
                        app.logger.info(f'Saving {file_key}: {file.filename}')
                        secure_name = secure_filename(file.filename)
                        filepath = os.path.join(temp_dir, secure_name)
                        file.save(filepath)
                        uploaded_files.append(filepath)
            
            if not uploaded_files:
                app.logger.error('No selected file')
                shutil.rmtree(temp_dir, ignore_errors=True)
                return 'No selected file', 400

            app.logger.info(f'Processing {len(uploaded_files)} files')
            analysis_results = []
            analyzer = VocalAnalyzer()  # Create single analyzer instance

            try:
                # Process each file
                for filepath in uploaded_files:
                    result = process_audio_file(filepath, analyzer)
                    if result:
                        app.logger.info(f'Adding result to analysis_results: {result}')
                        analysis_results.append(result)

                # Aggregate results from multiple files
                if analysis_results:
                    app.logger.info('Aggregating results')
                    final_analysis = aggregate_analyses(analysis_results)
                    app.logger.info(f'Final aggregated analysis: {final_analysis}')
                    
                    app.logger.info('Generating report')
                    artist_name = request.form.get('artist_name', '')
                    app.logger.info(f'Artist name: {artist_name}')
                    context = generate_report_context(final_analysis, artist_name)
                    app.logger.info(f'Generated context: {context}')
                    
                    if context:
                        app.logger.info('Processing complete')
                        return render_template('report_template.html', analysis=context)
                    else:
                        app.logger.error('Error generating report')
                        return 'Error generating report', 500
                else:
                    app.logger.error('No valid analysis results')
                    return 'No valid analysis results', 500

            except Exception as e:
                app.logger.error(f'Error processing files: {str(e)}\n{traceback.format_exc()}')
                return f'Error processing files: {str(e)}', 500
            finally:
                # Clean up temporary directory and all files in it
                shutil.rmtree(temp_dir, ignore_errors=True)
                gc.collect()  # Final garbage collection
            
        except Exception as e:
            app.logger.error(f'Error processing request: {str(e)}\n{traceback.format_exc()}')
            return f'Internal Server Error: {str(e)}', 500
    
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
    app.run(host='0.0.0.0', port=5000, debug=True)