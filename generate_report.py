from datetime import datetime
import numpy as np
from flask import Flask, render_template, redirect, url_for
from vocal_analyzer import VocalAnalyzer
from report_generator import ReportGenerator
import math

app = Flask(__name__)

def calculate_trends(analyses):
    """Calculate trends across multiple analyses."""
    trends = {}
    metrics = ['pitch_accuracy', 'breath_control', 'vibrato_quality', 
               'resonance_score', 'dynamic_range', 'articulation_score', 
               'emotional_expressivity']
    
    for metric in metrics:
        values = [analysis.get(metric, 0) for analysis in analyses]
        if len(values) > 1:
            current = values[-1]
            previous = values[-2]
            change = current - previous
            direction = 'positive' if change > 0 else 'negative' if change < 0 else 'neutral'
        else:
            current = values[0]
            change = 0
            direction = 'neutral'
        
        trends[metric] = {
            'current': round(current, 1),
            'change': round(change, 1),
            'direction': direction
        }
    
    return trends

def generate_comparative_insights(analyses):
    """Generate insights by comparing multiple performances."""
    insights = []
    
    # Compare pitch accuracy
    pitch_accuracies = [analysis.get('pitch_accuracy', 0) for analysis in analyses]
    if len(pitch_accuracies) > 1:
        if max(pitch_accuracies) - min(pitch_accuracies) > 1.0:
            insights.append({
                'title': 'Pitch Consistency',
                'description': 'Significant variation in pitch accuracy across performances. Consider focusing on consistent intonation practice.'
            })
    
    # Compare vocal range
    ranges = [analysis.get('vocal_range_semitones', 0) for analysis in analyses]
    if len(ranges) > 1:
        if max(ranges) - min(ranges) > 4:
            insights.append({
                'title': 'Range Development',
                'description': 'Notable variation in vocal range across performances. This may indicate inconsistent warm-up or technique.'
            })
    
    # Compare breath control
    breath_controls = [analysis.get('breath_control', 0) for analysis in analyses]
    if len(breath_controls) > 1:
        if max(breath_controls) - min(breath_controls) > 1.5:
            insights.append({
                'title': 'Breath Control Stability',
                'description': 'Breath control varies significantly between performances. Focus on consistent breathing technique.'
            })
    
    return insights

def generate_overall_recommendations(analyses):
    """Generate overall recommendations based on multiple performances."""
    recommendations = {
        'technique': [],
        'practice': [],
        'health': []
    }
    
    # Analyze pitch accuracy trends
    pitch_accuracies = [analysis.get('pitch_accuracy', 0) for analysis in analyses]
    if len(pitch_accuracies) > 1 and pitch_accuracies[-1] < 8.0:
        recommendations['technique'].append(
            'Focus on pitch accuracy exercises, particularly in challenging passages.'
        )
    
    # Analyze breath control trends
    breath_controls = [analysis.get('breath_control', 0) for analysis in analyses]
    if len(breath_controls) > 1 and breath_controls[-1] < 7.5:
        recommendations['practice'].append(
            'Incorporate daily breathing exercises to improve breath control and support.'
        )
    
    # Analyze vocal range consistency
    ranges = [analysis.get('vocal_range_semitones', 0) for analysis in analyses]
    if len(ranges) > 1 and max(ranges) - min(ranges) > 4:
        recommendations['technique'].append(
            'Work on consistent vocal warm-up routines to maintain range stability.'
        )
    
    # Analyze resonance trends
    resonance_scores = [analysis.get('resonance_score', 0) for analysis in analyses]
    if len(resonance_scores) > 1 and resonance_scores[-1] < 8.0:
        recommendations['technique'].append(
            'Focus on resonance exercises to improve tone quality and projection.'
        )
    
    return recommendations

def generate_report_context(analysis_results, artist_name):
    """Generate a context dictionary for the report template."""
    if not analysis_results:
        return None

    def normalize_score(score):
        """Normalize score to be between 0 and 10"""
        if score is None or not isinstance(score, (int, float)):
            return 0.0
        return min(max(float(score), 0.0), 10.0)

    # Calculate overall metrics
    pitch_accuracy = normalize_score(analysis_results.get('pitch_accuracy', 0))
    breath_control = normalize_score(analysis_results.get('breath_control', 0))
    vibrato_rate = float(analysis_results.get('vibrato_rate', 0))
    resonance = normalize_score(analysis_results.get('resonance', 0))
    dynamic_range = normalize_score(analysis_results.get('dynamic_range', 0))
    articulation = normalize_score(analysis_results.get('articulation', 0))

    # Calculate overall rating based on weighted average
    weights = {
        'pitch_accuracy': 0.3,
        'breath_control': 0.2,
        'resonance': 0.2,
        'dynamic_range': 0.15,
        'articulation': 0.15
    }
    
    overall_rating = sum([
        normalize_score(analysis_results.get(metric, 0)) * weight 
        for metric, weight in weights.items()
    ])

    # Industry averages (these could be stored in a configuration file)
    industry_averages = {
        'pitch_accuracy': 7.8,
        'breath_control': 7.4,
        'vibrato_quality': 7.6,
        'resonance': 7.3,
        'dynamic_range': 7.0
    }

    # Prepare technical assessment data
    technical_assessment = {
        'pitch_accuracy': {
            'score': pitch_accuracy,
            'industry_avg': industry_averages['pitch_accuracy'],
            'notes': 'Based on pitch stability and accuracy'
        },
        'breath_control': {
            'score': breath_control,
            'industry_avg': industry_averages['breath_control'],
            'notes': 'Based on phrase length and stability'
        },
        'vibrato_quality': {
            'score': normalize_score(analysis_results.get('vibrato_quality', 0)),
            'industry_avg': industry_averages['vibrato_quality'],
            'rate': vibrato_rate,
            'notes': 'Based on rate and depth consistency'
        },
        'resonance': {
            'score': resonance,
            'industry_avg': industry_averages['resonance'],
            'notes': 'Based on spectral balance and clarity'
        },
        'dynamic_range': {
            'score': dynamic_range,
            'industry_avg': industry_averages['dynamic_range'],
            'notes': 'Based on volume control and expression'
        }
    }

    # Determine areas for improvement (metrics below 7.5)
    improvement_areas = []
    for metric, data in technical_assessment.items():
        if data['score'] < 7.5:
            improvement_areas.append({
                'area': metric.replace('_', ' ').title(),
                'score': data['score'],
                'target': 8.0 if data['score'] < 6 else 7.5
            })

    # Determine strengths (metrics above 8.0)
    strengths = []
    for metric, data in technical_assessment.items():
        if data['score'] >= 8.0:
            strengths.append({
                'area': metric.replace('_', ' ').title(),
                'score': data['score']
            })

    # Add articulation if it's high enough
    if articulation >= 8.0:
        strengths.append({
            'area': 'Articulation',
            'score': articulation
        })

    # Prepare vocal range data
    vocal_range = {
        'lowest_note': analysis_results.get('lowest_note', 'Not Available'),
        'highest_note': analysis_results.get('highest_note', 'Not Available'),
        'range_span': analysis_results.get('range_span', '0 octaves')
    }

    # Prepare register transitions
    register_transitions = {
        'chest_to_mix_note': analysis_results.get('chest_to_mix', 'D4'),
        'mix_to_head_note': analysis_results.get('mix_to_head', 'E5'),
        'head_to_whistle_note': analysis_results.get('head_to_whistle', 'C6')
    }

    # Build the final context
    context = {
        'artist_name': artist_name,
        'date': datetime.now().strftime('%B %d, %Y'),
        'performances_analyzed': 1,  # This could be dynamic based on input
        'overall_rating': overall_rating,
        'technical_assessment': technical_assessment,
        'vocal_range': vocal_range,
        'register_transitions': register_transitions,
        'progress': {
            'consistency_score': normalize_score(analysis_results.get('consistency', 0)),
            'improvement_areas': improvement_areas,
            'strengths': strengths
        }
    }

    return context

def generate_sample_analysis():
    """Generate sample analysis data for testing."""
    return {
        'song_name': 'Sample Song',
        'duration': '3:45',
        'key': 'C Major',
        'date': datetime.now().strftime("%B %d, %Y %I:%M %p"),
        'pitch_accuracy': 9.2,
        'breath_control': 8.5,
        'vibrato_quality': 8.9,
        'resonance_score': 9.1,
        'dynamic_range': 8.8,
        'articulation_score': 8.5,
        'emotional_expressivity': 9.4,
        'vocal_range': 'E3 - G6',
        'range_position': 65,
        'lowest_note': 'E3',
        'highest_note': 'G6',
        'range_span': '3.2 octaves'
    }

@app.route('/')
def index():
    """Redirect root URL to test report."""
    return redirect(url_for('test_report'))

@app.route('/test_report')
def test_report():
    """Generate a test report with sample data."""
    analysis = generate_sample_analysis()
    context = generate_report_context(analysis, "Test Artist")
    return render_template('report_template.html', analysis=context)

if __name__ == '__main__':
    app.run(debug=True)