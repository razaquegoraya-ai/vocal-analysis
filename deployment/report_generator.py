import os
import weasyprint
from jinja2 import Template

class ReportGenerator:
    def __init__(self, template_path='templates/report_template.html'):
        self.template_path = template_path
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        if not os.path.exists(template_path):
            with open(template_path, 'w') as f:
                f.write(self._get_default_template())

    def _get_default_template(self):
        """Return the comprehensive report HTML template."""
        return """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Vocal Analysis Report: {{ artist_name }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
    .container { width: 90%; margin: 0 auto; padding: 20px; }
    .header { text-align: center; margin-bottom: 30px; }
    .title { font-size: 26pt; font-weight: bold; color: #333; }
    .subtitle { font-size: 16pt; color: #666; }
    .section { margin-bottom: 40px; }
    .section-title { font-size: 18pt; font-weight: bold; border-bottom: 2px solid #ccc; padding-bottom: 5px; margin-bottom: 20px; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; }
    th, td { border: 1px solid #ddd; padding: 8px; }
    th { background-color: #f2f2f2; }
    .metric { margin: 10px 0; }
    .metric-bar { background-color: #eee; border-radius: 5px; height: 20px; }
    .metric-fill { background-color: #4682B4; height: 100%; border-radius: 5px; }
    ul { list-style-type: disc; margin-left: 20px; }
    .footer { text-align: center; font-size: 10pt; color: #999; margin-top: 50px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="title">Vocal Analysis Report: {{ artist_name }}</div>
      <div class="subtitle">Date: {{ date }}</div>
      <div class="subtitle">Performances Analyzed: {{ performance_count }}</div>
    </div>

    <div class="section">
      <div class="section-title">Overview</div>
      <p>This report analyzes {{ performance_count }} vocal performance(s) using advanced acoustic analysis algorithms.</p>
      <p><strong>Overall Vocal Rating:</strong> {{ overall_rating }}/10</p>
      <div class="metric">
        <div class="metric-bar" style="width: 100%;">
          <div class="metric-fill" style="width: {{ overall_rating_percent }}%;"></div>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Technical Assessment</div>
      <table>
        <tr>
          <th>Parameter</th>
          <th>Score</th>
          <th>Industry Average</th>
          <th>Notes</th>
        </tr>
        {% for param in technical_params %}
        <tr>
          <td>{{ param.name }}</td>
          <td>{{ param.score }}/10</td>
          <td>{{ param.average }}/10</td>
          <td>{{ param.notes }}</td>
        </tr>
        {% endfor %}
      </table>
    </div>

    <div class="section">
      <div class="section-title">Vocal Range</div>
      <p>The analyzed performances demonstrate a vocal range of {{ vocal_range_octaves }} octaves, spanning from {{ min_pitch_hz }} Hz to {{ max_pitch_hz }} Hz.</p>
      <h4>Range Comparison to Standard Voice Types:</h4>
      <table>
        <tr>
          <th>Voice Type</th>
          <th>Standard Range (Hz)</th>
          <th>Exceeds Upper Limit by (Hz)</th>
        </tr>
        {% for voice_type, comp in range_comparisons.items() %}
        <tr>
          <td>{{ voice_type }}</td>
          <td>{{ comp.range_low_hz }} - {{ comp.range_high_hz }}</td>
          <td>{{ comp.exceeds_upper_by_hz }}</td>
        </tr>
        {% endfor %}
      </table>
      <p><strong>Range Classification:</strong> {{ range_classification }}</p>
    </div>

    <div class="section">
      <div class="section-title">Register Transition Points</div>
      <p><strong>Chest Voice to Mix:</strong> ~{{ register_transitions.chest_to_mix_note }} ({{ register_transitions.chest_to_mix_hz }} Hz)</p>
      <p><strong>Mix to Head Voice:</strong> ~{{ register_transitions.mix_to_head_note }} ({{ register_transitions.mix_to_head_hz }} Hz)</p>
      <p><strong>Head Voice to Whistle:</strong> ~{{ register_transitions.head_to_whistle_note }} ({{ register_transitions.head_to_whistle_hz }} Hz)</p>
    </div>

    <div class="section">
      <div class="section-title">Vocal Range Performance Metrics</div>
      <p><strong>Range Stability:</strong> {{ range_metrics.range_stability }}/10</p>
      <p><strong>Tonal Consistency Across Range:</strong> {{ range_metrics.tonal_consistency }}/10</p>
      <p><strong>Lower Register Power:</strong> {{ range_metrics.lower_register_power }}/10</p>
      <p><strong>Upper Register Clarity:</strong> {{ range_metrics.upper_register_clarity }}/10</p>
    </div>

    <div class="section">
      <div class="section-title">Stylistic Analysis</div>
      <p><strong>Vocal Texture:</strong> {{ stylistic_analysis.vocal_texture }}</p>
      <p><strong>Dynamic Range:</strong> {{ stylistic_analysis.dynamic_range }}</p>
      <p><strong>Articulation:</strong> {{ stylistic_analysis.articulation }}</p>
      <p><strong>Emotional Expressivity:</strong> {{ stylistic_analysis.emotional_expressivity }}</p>
      <p><strong>Genre Adaptability:</strong> {{ stylistic_analysis.genre_adaptability }}</p>
    </div>

    <div class="section">
      <div class="section-title">Strengths & Development Areas</div>
      <div style="display: flex;">
        <div style="flex: 1;">
          <h4>Key Strengths</h4>
          <ul>
            {% for s in strengths %}
            <li>{{ s }}</li>
            {% endfor %}
          </ul>
        </div>
        <div style="flex: 1;">
          <h4>Development Areas</h4>
          <ul>
            {% for d in development_areas %}
            <li>{{ d }}</li>
            {% endfor %}
          </ul>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">Vocal Health Observations</div>
      <p>{{ vocal_health }}</p>
    </div>

    <div class="footer">
      <p>Analysis conducted using VocalMetrics™ AI 4.0</p>
      <p>Report generated on {{ date }}</p>
    </div>
  </div>
</body>
</html>"""

    def generate_html_report(self, context, output_html='reports/final_report.html'):
        with open(self.template_path, 'r') as f:
            template = Template(f.read())
        html_content = template.render(context)
        os.makedirs(os.path.dirname(output_html), exist_ok=True)
        with open(output_html, 'w') as f:
            f.write(html_content)
        return output_html

    def generate_pdf_report(self, html_path, output_pdf='reports/final_report.pdf'):
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
        weasyprint.HTML(html_path).write_pdf(output_pdf)
        return output_pdf

# Usage example:
if __name__ == "__main__":
    context = {
      "artist_name": "Sarah Mitchell",
      "date": "March 14, 2025 10:00 AM",
      "performance_count": 4,
      "overall_rating": 8.7,
      "overall_rating_percent": 87,
      "technical_params": [],
      "vocal_range_octaves": 3.2,
      "min_pitch_hz": 164.81,
      "max_pitch_hz": 1567.98,
      "range_comparisons": {},
      "range_classification": "",
      "register_transitions": {
          "chest_to_mix_hz": 293.66,
          "chest_to_mix_note": "D4",
          "mix_to_head_hz": 659.26,
          "mix_to_head_note": "E5",
          "head_to_whistle_hz": 1046.50,
          "head_to_whistle_note": "C6"
      },
      "range_metrics": {
          "range_stability": 8.6,
          "tonal_consistency": 8.3,
          "lower_register_power": 7.9,
          "upper_register_clarity": 9.2
      },
      "stylistic_analysis": {
          "vocal_texture": "Mezzo-soprano with rich, bright, warm, and resonant qualities",
          "dynamic_range": "8.8/10 - Effective contrast between pianissimo and fortissimo",
          "articulation": "8.6/10 - Clear diction with minimal consonant distortion",
          "emotional_expressivity": "9.4/10 - Exceptional vocal coloration to convey emotion",
          "genre_adaptability": "8.2/10 - Strong in pop/soul with potential in adjacent genres"
      },
      "strengths": [
          "Exceptional pitch accuracy and intonation control",
          "Rich vocal resonance creating a full, projected tone",
          "Exceptional ability to convey emotion through vocal coloration"
      ],
      "development_areas": [
          "Expand dynamic control for more dramatic contrast",
          "Develop more consistent resonance across entire range",
          "Expand head voice/mix transitions for smoother upper register access"
      ],
      "vocal_health": "Spectrogram analysis indicates healthy vocal fold function with no signs of nodules or significant fatigue patterns. Slight tension observed in the D5-F5 range during sustained passages."
    }
    rg = ReportGenerator()
    html_report = rg.generate_html_report(context)
    pdf_report = rg.generate_pdf_report(html_report)
    print(f"HTML report: {html_report}")
    print(f"PDF report: {pdf_report}")