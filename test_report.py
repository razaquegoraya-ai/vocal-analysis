from flask import Flask, render_template, redirect, url_for
import os
from datetime import datetime

app = Flask(__name__)

def generate_sample_analysis():
    """Generate comprehensive sample analysis data for testing."""
    return {
        'artist_name': 'Sarah Mitchell',
        'date': 'March 14, 2023',
        'performances_analyzed': 4,
        'filename': 'vocal_performance.mp3',
        
        # Overall Rating
        'overall_rating': 8.7,
        'overall_rating_note': 'The visual representation would typically show an 87% filled bar.',
        
        # Technical Assessment
        'technical_assessment': {
            'pitch_accuracy': {
                'score': 9.2,
                'industry_avg': 7.8,
                'notes': 'Exceptional intonation with 92.0% accuracy'
            },
            'vocal_range': {
                'score': 8.4,
                'industry_avg': 7.0,
                'notes': 'Wide range with good tone control across registers'
            },
            'breath_control': {
                'score': 8.5,
                'industry_avg': 7.4,
                'notes': 'Extended phrases maintained with minimal strain'
            },
            'vibrato_quality': {
                'score': 8.9,
                'industry_avg': 7.6,
                'notes': 'Natural, controlled oscillation at 5.8Hz'
            },
            'resonance': {
                'score': 9.1,
                'industry_avg': 7.3,
                'notes': 'Strong forward placement with balanced overtones'
            }
        },
        
        # Vocal Range Analysis
        'vocal_range': {
            'octaves': 3.2,
            'lowest_note': 'E3',
            'lowest_hz': 164.81,
            'highest_note': 'G6',
            'highest_hz': 1567.98,
            
            # Voice Type Ranges
            'voice_ranges': {
                'bass': {'range': 'C2 – E4', 'hz': '65.41 – 329.63', 'note': 'Subject exceeds upper limit by 1238.35 Hz'},
                'baritone': {'range': 'F2 – G4', 'hz': '87.31 – 392.00', 'note': 'Subject exceeds upper limit by 1175.98 Hz'},
                'tenor': {'range': 'C3 – C5', 'hz': '130.81 – 523.25', 'note': 'Subject exceeds upper limit by 1044.73 Hz'},
                'alto': {'range': 'F3 – F5', 'hz': '174.61 – 698.46', 'note': 'Subject exceeds upper limit by 869.52 Hz'},
                'mezzo_soprano': {'range': 'G3 – A5', 'hz': '196.00 – 880.00', 'note': 'Subject exceeds upper limit by 687.98 Hz'},
                'soprano': {'range': 'C4 – C6', 'hz': '261.63 – 1046.50', 'note': 'Subject exceeds upper limit by 521.48 Hz'}
            },
            'classification': 'Mezzo-Soprano with extended Soprano range'
        },
        
        # Register Transitions
        'register_transitions': {
            'chest_to_mix': {'note': 'D4', 'hz': 293.66},
            'mix_to_head': {'note': 'E5', 'hz': 659.26},
            'head_to_whistle': {'note': 'C6', 'hz': 1046.50}
        },
        
        # Performance Metrics
        'performance_metrics': {
            'range_stability': 8.6,
            'tonal_consistency': 8.3,
            'lower_register_power': 7.9,
            'upper_register_clarity': 9.2
        },
        
        # Additional Metrics
        'dynamic_range': {'score': 8.8, 'note': 'Effective contrast between pianissimo and fortissimo'},
        'articulation': {'score': 8.5, 'note': 'Clear diction with minimal consonant distortion'},
        'emotional_expressivity': {'score': 9.4, 'note': 'Exceptional vocal coloration to convey emotion'},
        'genre_adaptability': {'score': 8.2, 'note': 'Strong in pop/rock with potential in adjacent genres'},
        
        # Analysis Sections
        'strengths': [
            'Exceptional pitch accuracy and intonation control',
            'Rich vocal resonance creating a full, projected tone',
            'Outstanding ability to convey emotion through vocal coloration'
        ],
        'development_areas': [
            'Expand dynamic control for more dramatic contrast',
            'Develop more consistent resonance across the entire range',
            'Improve head voice/mix transitions for smoother upper register access'
        ],
        'vocal_health': {
            'observations': 'Spectrogram analysis indicates healthy vocal fold function with no signs of nodules or significant fatigue patterns. A slight tension is observed in the D5-F5 range during sustained passages.'
        },
        'stylistic_analysis': {
            'vocal_texture': 'Mezzo-soprano with rich, bright, warm, and resonant qualities'
        },
        
        # Meta Information
        'analysis_version': 'VocalMetrics™ A14.0',
        'report_date': 'March 14, 2023'
    }

@app.route('/')
def index():
    """Redirect root URL to test report."""
    return redirect(url_for('test_report'))

@app.route('/test_report')
def test_report():
    """Generate a test report with comprehensive sample data."""
    analysis = generate_sample_analysis()
    return render_template('report_template.html', analysis=analysis)

if __name__ == '__main__':
    app.run(debug=True, port=5001) 