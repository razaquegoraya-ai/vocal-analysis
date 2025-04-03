from datetime import datetime
import numpy as np
import math

def calculate_trends(analyses):
    """Calculate trends across multiple analyses."""
    trends = {}
    metrics = ['pitch_accuracy', 'breath_control', 'vibrato_quality', 
               'resonance', 'dynamic_range']
    
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
    resonance_scores = [analysis.get('resonance', 0) for analysis in analyses]
    if len(resonance_scores) > 1 and resonance_scores[-1] < 8.0:
        recommendations['technique'].append(
            'Focus on resonance exercises to improve tone quality and projection.'
        )
    
    return recommendations

def generate_report_context(analysis_results, artist_name, file_number=None):
    """Generate a context dictionary for the report template."""
    if not analysis_results:
        return None

    def normalize_score(score):
        """Normalize score to be between 0 and 10"""
        if score is None or not isinstance(score, (int, float)):
            return 0.0
        return min(max(float(score), 0.0), 10.0)

    # Map the metrics from VocalAnalyzer to report metrics
    metrics_mapping = {
        'pitch_accuracy': 'pitch_accuracy',
        'breath_control': 'breath_control',
        'resonance': 'resonance',
        'dynamic_range': 'dynamic_range',
        'vibrato_rate': 'vibrato_rate',
        'vibrato_extent': 'vibrato_extent'
    }

    # Initialize context with default values
    context = {
        'artist_name': artist_name,
        'date': datetime.now().strftime('%B %d, %Y'),
        'file_number': file_number,
        'overall_rating': 0.0,
        'consistency_score': 0.0,
        'pitch_accuracy': 0.0,
        'breath_control': 0.0,
        'vibrato_rate': 0.0,
        'vibrato_extent': 0.0,
        'resonance': 0.0,
        'dynamic_range': 0.0,
        'lowest_note': 'Not Available',
        'highest_note': 'Not Available',
        'range_span': '0 octaves',
        'chest_to_mix': 'D4',
        'mix_to_head': 'E5',
        'head_to_whistle': 'C6'
    }

    # Update context with actual values from analysis_results
    for output_key, input_key in metrics_mapping.items():
        if input_key in analysis_results:
            value = analysis_results[input_key]
            if isinstance(value, (int, float)):
                context[output_key] = normalize_score(value)
            else:
                context[output_key] = value

    # Copy non-numeric values
    if 'lowest_note' in analysis_results:
        context['lowest_note'] = analysis_results['lowest_note']
    if 'highest_note' in analysis_results:
        context['highest_note'] = analysis_results['highest_note']
    if 'range_span' in analysis_results:
        context['range_span'] = analysis_results['range_span']
    if 'chest_to_mix' in analysis_results:
        context['chest_to_mix'] = analysis_results['chest_to_mix']
    if 'mix_to_head' in analysis_results:
        context['mix_to_head'] = analysis_results['mix_to_head']
    if 'head_to_whistle' in analysis_results:
        context['head_to_whistle'] = analysis_results['head_to_whistle']

    # Calculate overall rating based on weighted average
    weights = {
        'pitch_accuracy': 0.3,
        'breath_control': 0.2,
        'resonance': 0.2,
        'dynamic_range': 0.15,
        'vibrato_quality': 0.15
    }

    overall_rating = 0.0
    total_weight = 0.0
    for metric, weight in weights.items():
        if metric in context and isinstance(context[metric], (int, float)):
            overall_rating += context[metric] * weight
            total_weight += weight

    if total_weight > 0:
        context['overall_rating'] = overall_rating / total_weight

    # Calculate consistency score based on standard deviation of metrics
    metrics_for_consistency = ['pitch_accuracy', 'breath_control', 'resonance', 'dynamic_range']
    values = [context[metric] for metric in metrics_for_consistency if metric in context]
    if values:
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        # Convert to consistency score (inverse of standard deviation)
        context['consistency_score'] = max(0, 10 - std_dev)

    # Industry averages
    context['industry_averages'] = {
        'pitch_accuracy': 7.8,
        'breath_control': 7.4,
        'vibrato_quality': 7.6,
        'resonance': 7.3,
        'dynamic_range': 7.0
    }

    # Calculate vibrato quality score
    if 'vibrato_rate' in context and 'vibrato_extent' in context:
        vibrato_quality = 0.0
        if 4.5 <= context['vibrato_rate'] <= 6.5:  # Ideal vibrato rate range
            rate_score = 10.0 - abs(5.5 - context['vibrato_rate'])
            extent_score = min(10.0, context['vibrato_extent'] * 2)
            vibrato_quality = (rate_score + extent_score) / 2
        context['vibrato_quality'] = vibrato_quality

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
        'resonance': 9.1,
        'dynamic_range': 8.8,
        'vocal_range': 'E3 - G6',
        'range_position': 65,
        'lowest_note': 'E3',
        'highest_note': 'G6',
        'range_span': '3.2 octaves'
    }