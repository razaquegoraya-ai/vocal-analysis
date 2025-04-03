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
    if not analyses or len(analyses) < 2:
        return []

    insights = []
    
    # Compare pitch accuracy trends
    pitch_accuracies = [a.get('pitch_accuracy', 0) for a in analyses]
    if max(pitch_accuracies) - min(pitch_accuracies) > 1.0:
        insights.append({
            'title': 'Pitch Consistency',
            'description': 'Significant variation in pitch accuracy across performances. Consider focusing on consistent intonation practice.'
        })
    
    # Compare breath control
    breath_controls = [a.get('breath_control', 0) for a in analyses]
    if max(breath_controls) - min(breath_controls) > 1.5:
        insights.append({
            'title': 'Breath Control Development',
            'description': 'Breath control shows notable variation. Focus on maintaining consistent support across performances.'
        })
    
    # Compare resonance
    resonance_scores = [a.get('resonance', 0) for a in analyses]
    if max(resonance_scores) - min(resonance_scores) > 1.0:
        insights.append({
            'title': 'Resonance Consistency',
            'description': 'Resonance quality varies between performances. Work on maintaining consistent tone production.'
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

def generate_report_context(analysis_results, artist_name, performance_count=1):
    """Generate a context dictionary for the report template."""
    if not analysis_results:
        return None

    def normalize_score(score):
        """Normalize score to be between 0 and 10"""
        if score is None or not isinstance(score, (int, float)):
            return 0.0
        return min(max(float(score), 0.0), 10.0)

    # Initialize context with basic information
    context = {
        'artist_name': artist_name,
        'date': datetime.now().strftime('%B %d, %Y'),
        'performances_analyzed': performance_count,
        'version': 'AI 4.0',
        
        # Overall metrics
        'overall_rating': 0.0,
        'consistency_score': 0.0,
        
        # Technical metrics
        'pitch_accuracy': analysis_results.get('pitch_accuracy', 0.0),
        'breath_control': analysis_results.get('breath_control', 0.0),
        'resonance': analysis_results.get('resonance', 0.0),
        'dynamic_range': analysis_results.get('dynamic_range', 0.0),
        'vibrato_rate': analysis_results.get('vibrato_rate', 0.0),
        'vibrato_extent': analysis_results.get('vibrato_extent', 0.0),
        
        # Range information
        'lowest_note': analysis_results.get('lowest_note', 'N/A'),
        'highest_note': analysis_results.get('highest_note', 'N/A'),
        'range_span': analysis_results.get('range_span', '0 octaves'),
        'range_classification': analysis_results.get('voice_classification', 'Undetermined'),
        'range_classification_notes': analysis_results.get('range_classification_notes', ''),
        
        # Register transitions
        'chest_to_mix': analysis_results.get('chest_to_mix', 'D4'),
        'mix_to_head': analysis_results.get('mix_to_head', 'E5'),
        'head_to_whistle': analysis_results.get('head_to_whistle', 'C6'),
        
        # Performance metrics
        'range_stability': analysis_results.get('range_stability', 5.0),
        'tonal_consistency': analysis_results.get('tonal_consistency', 5.0),
        'lower_register_power': analysis_results.get('lower_register_power', 5.0),
        'upper_register_clarity': analysis_results.get('upper_register_clarity', 5.0),
        
        # Stylistic analysis
        'vocal_texture': analysis_results.get('vocal_texture', ''),
        'dynamic_range_description': analysis_results.get('dynamic_range_description', ''),
        'articulation_description': analysis_results.get('articulation_description', ''),
        'emotional_expressivity_description': analysis_results.get('emotional_expressivity_description', ''),
        'genre_adaptability_description': analysis_results.get('genre_adaptability_description', ''),
        
        # Strengths and development
        'strengths': analysis_results.get('strengths', []),
        'development_areas': analysis_results.get('development_areas', []),
        
        # Health observations
        'vocal_health_observations': analysis_results.get('vocal_health_observations', ''),
        
        # Metric notes
        'pitch_accuracy_notes': analysis_results.get('pitch_accuracy_notes', ''),
        'breath_control_notes': analysis_results.get('breath_control_notes', ''),
        'resonance_notes': analysis_results.get('resonance_notes', ''),
        'vocal_range_notes': analysis_results.get('vocal_range_notes', ''),
        
        # Industry averages
        'industry_averages': {
            'pitch_accuracy': 7.8,
            'breath_control': 7.4,
            'resonance': 7.3,
            'dynamic_range': 7.0,
            'vocal_range': 7.5
        }
    }

    # Calculate overall rating based on weighted average of key metrics
    weights = {
        'pitch_accuracy': 0.25,
        'breath_control': 0.20,
        'resonance': 0.20,
        'dynamic_range': 0.15,
        'range_stability': 0.10,
        'tonal_consistency': 0.10
    }

    weighted_sum = 0
    total_weight = 0
    for metric, weight in weights.items():
        if metric in context and isinstance(context[metric], (int, float)):
            weighted_sum += context[metric] * weight
            total_weight += weight

    if total_weight > 0:
        context['overall_rating'] = weighted_sum / total_weight

    # Calculate consistency score
    consistency_metrics = ['range_stability', 'tonal_consistency']
    consistency_values = [context[m] for m in consistency_metrics if isinstance(context[m], (int, float))]
    if consistency_values:
        context['consistency_score'] = sum(consistency_values) / len(consistency_values)

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

def generate_development_plan(analysis):
    """Generate a personalized development plan based on analysis results."""
    plan = {
        'short_term': [],
        'medium_term': [],
        'long_term': []
    }
    
    # Short-term goals (immediate focus areas)
    if analysis.get('pitch_accuracy', 0) < 7.0:
        plan['short_term'].append('Daily pitch accuracy exercises focusing on intervals and scales')
    if analysis.get('breath_control', 0) < 7.0:
        plan['short_term'].append('Implement structured breathing exercises into daily practice routine')
    
    # Medium-term goals (1-3 months)
    if analysis.get('resonance', 0) < 8.0:
        plan['medium_term'].append('Work with exercises to develop consistent resonance across range')
    if analysis.get('dynamic_range', 0) < 8.0:
        plan['medium_term'].append('Practice contrasting dynamic levels in exercises and repertoire')
    
    # Long-term goals (3+ months)
    if analysis.get('range_stability', 0) < 8.0:
        plan['long_term'].append('Gradually expand range while maintaining consistent tone quality')
    if analysis.get('tonal_consistency', 0) < 8.0:
        plan['long_term'].append('Focus on achieving uniform tone quality across all registers')
    
    return plan