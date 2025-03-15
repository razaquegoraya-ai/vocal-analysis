# Vocal Analysis System

A comprehensive vocal analysis system that evaluates singing performances using advanced audio processing techniques and machine learning. The system analyzes various aspects of vocal performance including pitch accuracy, breath control, vibrato quality, resonance, and more.

## Features

- **Vocal Separation**: Isolates vocals from background music using Spleeter
- **Technical Analysis**: 
  - Pitch accuracy and stability
  - Breath control assessment
  - Vibrato rate and extent
  - Resonance and timbre analysis
  - Dynamic range evaluation
  - Articulation scoring
- **Range Analysis**:
  - Vocal range detection
  - Register transition points
  - Voice type classification
- **Report Generation**:
  - Detailed HTML reports
  - PDF export capability
  - Visual metrics and graphs
  - Professional formatting

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
venv\Scripts\activate     # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: For M1/M2 Mac users, the system will automatically install tensorflow-macos instead of the standard tensorflow package.

## Usage

### Basic Usage

```python
from vocal_analyzer import VocalAnalyzer
from report_generator import ReportGenerator

# Initialize the analyzer
analyzer = VocalAnalyzer()

# Analyze an audio file
analysis = analyzer.analyze_file('path/to/audio.mp3')

# Generate a report
context = generate_report_context(analysis, "Artist Name", 1)
rg = ReportGenerator()
html_report = rg.generate_html_report(context)
pdf_report = rg.generate_pdf_report(html_report)
```

### Command Line Usage

```bash
python generate_report.py path/to/audio.mp3
```

## System Requirements

- Python 3.8 or higher
- FFmpeg (for audio processing)
- At least 4GB RAM
- Sufficient disk space for temporary files during vocal separation

## Project Structure

- `vocal_analyzer.py`: Core analysis engine
- `vocal_separator.py`: Handles vocal isolation
- `report_generator.py`: Generates HTML/PDF reports
- `generate_report.py`: Main script for report generation
- `templates/`: Contains HTML templates for reports
- `models/`: Directory for ML models (created on first run)
- `reports/`: Output directory for generated reports

## Dependencies

Major dependencies include:
- librosa: Audio processing
- tensorflow/tensorflow-macos: Machine learning
- essentia: Audio feature extraction
- spleeter: Vocal separation
- weasyprint: PDF generation
- jinja2: HTML templating

## Output

The system generates two types of reports:
1. HTML Report: Interactive web-based report with visualizations
2. PDF Report: Print-friendly version of the analysis

Reports include:
- Overall vocal rating
- Technical assessment metrics
- Vocal range analysis
- Register transition points
- Stylistic analysis
- Strengths and development areas
- Vocal health observations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Spleeter by Deezer Research
- Essentia by Music Technology Group
- librosa development team 