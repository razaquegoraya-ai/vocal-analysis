import os
import tempfile
import shutil
import numpy as np
import librosa
import soundfile as sf
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

class VocalSeparator:
    """
    Vocal separator using Demucs (state-of-the-art music source separation).
    """
    def __init__(self, model_name='htdemucs', device=None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp_dir = tempfile.mkdtemp(prefix="vocal_sep_")
        self._initialize_model()

    def _initialize_model(self):
        try:
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            print(f"Initialized Demucs model '{self.model_name}' on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Demucs model: {str(e)}")

    def separate(self, audio, sr):
        """Separate vocals using Demucs."""
        try:
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)

            # Demucs expects input as: (batch, channels, time)
            audio_tensor = torch.tensor(audio).float()
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)

            # Apply the model
            with torch.no_grad():
                sources = apply_model(self.model, audio_tensor, device=self.device)[0]
                # sources will be a tensor of shape (sources, channels, time)
                vocals = sources[self.model.sources.index('vocals')].squeeze().cpu().numpy()

            return vocals

        except Exception as e:
            raise RuntimeError(f"Vocal separation failed: {str(e)}")

    def cleanup(self):
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Warning: Could not remove temporary directory: {str(e)}")

    def __del__(self):
        self.cleanup()

# Usage example:
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} input_audio.mp3 output_vocals.wav")
        exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    try:
        separator = VocalSeparator()
        audio, sr = librosa.load(input_path, sr=None)
        print(f"Separating vocals from {input_path}...")
        vocals = separator.separate(audio, sr)
        sf.write(output_path, vocals, sr)
        print(f"Saved vocals to {output_path}")
        separator.cleanup()
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)