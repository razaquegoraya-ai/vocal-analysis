# Vocal Analysis Project

<img width="1440" alt="image" src="https://github.com/user-attachments/assets/49e7f1e7-a6ba-43a1-8b14-264be3a14ba0" />


A web application for analyzing vocal performances using advanced audio processing techniques.

## Features

- Vocal separation using Demucs
- Detailed vocal analysis including:
  - Pitch accuracy
  - Breath control
  - Vibrato quality
  - Resonance
  - Dynamic range
  - Articulation
  - Emotional expressivity
  - Vocal range and register transitions
- Interactive HTML reports
- PDF export capability

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vocal-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to: `http://localhost:5000`

3. Upload an audio file (supported formats: MP3, WAV, M4A, AAC, OGG)

4. Click "Analyze Voice" to generate a detailed report

## Project Structure

- `app.py`: Main Flask application
- `vocal_analyzer.py`: Core vocal analysis functionality
- `vocal_separator.py`: Audio separation using Demucs
- `templates/`: HTML templates for the web interface
- `uploads/`: Directory for temporary audio file storage

## Requirements

- Python 3.9+
- See `requirements.txt` for full list of dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Spleeter by Deezer Research
- Essentia by Music Technology Group
- librosa development team 
