import os
import tempfile
import shutil
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

class VocalSeparator:
    """
    Vocal separator using Demucs (state-of-the-art music source separation).
    """
    def __init__(self):
        """Initialize the vocal separator with Demucs."""
        self.model = get_model('htdemucs')
        self.model.cpu()
        self.model.eval()
        self.temp_dir = None

    def separate(self, audio, sr):
        """Separate vocals from the audio using Demucs.
        
        Args:
            audio: Audio array (channels, samples) or (samples,)
            sr: Sample rate
            
        Returns:
            Separated vocals as numpy array
        """
        try:
            # Create temporary directory for processing
            self.temp_dir = tempfile.mkdtemp(prefix='vocal_sep_')
            
            # Ensure audio is in the correct format (channels, samples)
            if len(audio.shape) == 1:
                audio = np.stack([audio, audio])
            elif audio.shape[0] > 2:
                audio = audio[:2]  # Take first two channels if more than stereo
            
            # Process in chunks to save memory
            chunk_size = sr * 30  # Process 30 seconds at a time
            hop_size = sr * 2     # 2 seconds overlap
            
            vocals_list = []
            for i in range(0, audio.shape[1], chunk_size - hop_size):
                chunk = audio[:, i:i + chunk_size]
                if chunk.shape[1] < chunk_size:
                    # Pad last chunk if needed
                    pad_size = chunk_size - chunk.shape[1]
                    chunk = np.pad(chunk, ((0, 0), (0, pad_size)))
                
                # Convert to torch tensor
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
                chunk_tensor = chunk_tensor.unsqueeze(0)
                
                # Apply the model
                with torch.no_grad():
                    sources = apply_model(self.model, chunk_tensor, device='cpu', progress=False)
                    
                # Extract vocals
                chunk_vocals = sources[0, self.model.sources.index('vocals')].numpy()
                
                # Remove padding from last chunk
                if i + chunk_size > audio.shape[1]:
                    chunk_vocals = chunk_vocals[:, :-(i + chunk_size - audio.shape[1])]
                
                vocals_list.append(chunk_vocals)
            
            # Combine chunks with crossfade
            vocals = np.zeros((2, audio.shape[1]))
            overlap = np.linspace(0, 1, hop_size)
            
            current_pos = 0
            for i, chunk in enumerate(vocals_list):
                if i == 0:
                    vocals[:, :chunk.shape[1]] = chunk
                else:
                    # Crossfade
                    vocals[:, current_pos:current_pos + hop_size] *= (1 - overlap)
                    chunk[:, :hop_size] *= overlap
                    vocals[:, current_pos:current_pos + chunk.shape[1]] += chunk
                current_pos += chunk_size - hop_size
            
            return vocals
            
        except Exception as e:
            print(f"Error in vocal separation: {str(e)}")
            # Fall back to basic separation
            try:
                y_mono = np.mean(audio, axis=0) if len(audio.shape) > 1 else audio
                y_harmonic, _ = librosa.effects.hpss(y_mono)
                return np.stack([y_harmonic, y_harmonic])
            except:
                return np.array([])
        
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Could not remove temporary directory: {str(e)}")
        self.temp_dir = None

    def __del__(self):
        """Ensure cleanup on object destruction."""
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
        audio, sr = librosa.load(input_path, sr=None, mono=False)  # Load as stereo
        print(f"Separating vocals from {input_path}...")
        vocals = separator.separate(audio, sr)
        sf.write(output_path, vocals, sr)
        print(f"Saved vocals to {output_path}")
        separator.cleanup()
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)