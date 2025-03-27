import librosa
import numpy as np
from scipy.stats import pearsonr
from scipy import signal
import tensorflow as tf
from pathlib import Path
import json
import math

class VocalAnalyzer:
    def __init__(self, model_path='models/vocal_classifier.h5'):
        self.model = None
        # Try to load ML model if available; otherwise, run basic analysis.
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Loaded model from {model_path}.")
        except Exception as e:
            print(f"Model not found at {model_path}. Running without ML enhancement.")
            print("For better performance, obtain a pre-trained model and place it in the 'models' folder.")
        self.features_cache = {}

    def analyze_file(self, file_path):
        """Analyze a single audio file and extract vocal features."""
        try:
            # Load audio as stereo for better vocal separation
            y, sr = librosa.load(file_path, sr=44100, mono=False)
            if len(y.shape) == 1:  # If still mono, convert to stereo
                y = np.stack([y, y])
        except Exception as e:
            return {"error": f"Failed to load audio file: {str(e)}"}
        
        # First attempt with Demucs
        vocals = self._extract_vocals(y, sr)
        
        # Check for vocal content using multiple methods
        has_vocals = False
        
        # Method 1: Energy in vocal frequency range
        if len(vocals) > 0:
            spec = np.abs(librosa.stft(np.mean(vocals, axis=0) if len(vocals.shape) > 1 else vocals))
            freqs = librosa.fft_frequencies(sr=sr)
            
            # Focus on typical vocal frequency range (80Hz - 1100Hz)
            vocal_range_mask = (freqs >= 80) & (freqs <= 1100)
            vocal_range_energy = np.sum(spec[vocal_range_mask])
            total_energy = np.sum(spec)
            
            if vocal_range_energy / (total_energy + 1e-10) >= 0.15:  # Lowered threshold
                has_vocals = True
        
        # Method 2: Check for periodic patterns typical of speech/singing
        if not has_vocals and len(vocals) > 0:
            # Use autocorrelation to detect periodicity
            y_mono = np.mean(vocals, axis=0) if len(vocals.shape) > 1 else vocals
            hop_length = 512
            oenv = librosa.onset.onset_strength(y=y_mono, sr=sr, hop_length=hop_length)
            tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
            
            # Check if there are clear periodic patterns
            if np.max(tempogram) > 0.5:  # Lowered threshold
                has_vocals = True
        
        # Method 3: Spectral rolloff and contrast
        if not has_vocals and len(vocals) > 0:
            y_mono = np.mean(vocals, axis=0) if len(vocals.shape) > 1 else vocals
            
            # Spectral rolloff should be high for vocals
            rolloff = librosa.feature.spectral_rolloff(y=y_mono, sr=sr)
            
            # Spectral contrast should show clear peaks in vocal frequencies
            contrast = librosa.feature.spectral_contrast(y=y_mono, sr=sr)
            
            if np.mean(rolloff) > sr/4 or np.max(contrast) > 20:  # Adjusted thresholds
                has_vocals = True
        
        if not has_vocals:
            return {"error": "No vocal content detected"}
        
        # Extract features from the detected vocals
        features = self._extract_features(vocals, sr)
        self.features_cache[Path(file_path).name] = features
        
        # Add duration information
        duration_sec = len(y[0]) / sr
        minutes = int(duration_sec // 60)
        seconds = int(duration_sec % 60)
        features['duration'] = f"{minutes}:{seconds:02d}"
        
        # Estimate musical key
        key = self._estimate_musical_key(vocals, sr)
        features['key'] = key
        
        return features

    def _freq_to_bin(self, freq, sr):
        """Convert frequency to FFT bin number."""
        n_fft = 2048  # Default FFT size
        return int(freq * n_fft / sr)

    def _extract_vocals(self, y, sr):
        """Extract vocal segments using the VocalSeparator."""
        from vocal_separator import VocalSeparator
        try:
            separator = VocalSeparator()
            vocals = separator.separate(y, sr)
            separator.cleanup()
            
            # Additional post-processing to clean up the vocals
            if len(vocals) > 0:
                # Apply a bandpass filter to focus on vocal frequencies
                vocals = self._apply_bandpass_filter(vocals, sr)
            return vocals
        except Exception as e:
            print(f"Advanced vocal separation failed: {str(e)}. Falling back to basic separation.")
            
        # Enhanced fallback method
        try:
            y_mono = np.mean(y, axis=0) if len(y.shape) > 1 else y
            y_harmonic, _ = librosa.effects.hpss(y_mono)
            
            # Use spectral contrast to identify vocal regions
            S = np.abs(librosa.stft(y_harmonic))
            contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
            contrast_mean = np.mean(contrast, axis=1)
            
            # Create a mask for vocal frequencies
            freqs = librosa.fft_frequencies(sr=sr)
            vocal_mask = (freqs >= 200) & (freqs <= 3000)
            
            # Apply the mask
            S_filtered = S * vocal_mask.reshape(-1, 1)
            vocals = librosa.istft(S_filtered)
            
            # Apply bandpass filter
            vocals = self._apply_bandpass_filter(vocals, sr)
            
            return vocals
        except Exception as e:
            print(f"Basic separation also failed: {str(e)}")
            return np.array([])

    def _apply_bandpass_filter(self, audio, sr):
        """Apply bandpass filter to focus on vocal frequencies."""
        # Design bandpass filter
        nyquist = sr / 2
        low = 200 / nyquist
        high = 3000 / nyquist
        b, a = signal.butter(5, [low, high], btype='band')
        
        # Apply filter
        return signal.filtfilt(b, a, audio)

    def _extract_features(self, vocals, sr):
        """Extract core vocal features and compute additional metrics."""
        # Convert to mono for feature extraction if needed
        y_mono = np.mean(vocals, axis=0) if len(vocals.shape) > 1 else vocals
        
        # Extract pitch and confidence using more robust method
        pitches, magnitudes = librosa.piptrack(y=y_mono, sr=sr, fmin=50, fmax=2000)
        
        # Use magnitude-weighted statistics for more accurate pitch estimation
        pitch_mask = magnitudes > np.median(magnitudes) * 0.1
        valid_pitches = pitches[pitch_mask]
        valid_magnitudes = magnitudes[pitch_mask]
        
        if len(valid_pitches) == 0:
            return {
                "error": "Could not extract reliable pitch information"
            }
        
        # Calculate weighted statistics
        pitch_mean = np.average(valid_pitches, weights=valid_magnitudes)
        pitch_std = np.sqrt(np.average((valid_pitches - pitch_mean)**2, weights=valid_magnitudes))
        
        # Get min/max pitches (weighted by magnitude)
        min_pitch = np.min(valid_pitches[valid_magnitudes > np.max(valid_magnitudes) * 0.1])
        max_pitch = np.max(valid_pitches[valid_magnitudes > np.max(valid_magnitudes) * 0.1])
        
        # Calculate vocal range in semitones
        vocal_range_semitones = 12 * np.log2(max_pitch/min_pitch) if min_pitch > 0 else 0
        
        # Analyze vibrato
        vibrato_rate, vibrato_extent = self._analyze_vibrato(valid_pitches)
        
        # Extract MFCCs for timbre analysis
        mfccs = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        
        # Calculate spectral features
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_mono, sr=sr)))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y_mono, sr=sr)))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y_mono, sr=sr), axis=1).tolist()
        
        # Calculate dynamic range
        rms = librosa.feature.rms(y=y_mono)[0]
        dynamic_range = float(np.percentile(rms, 95) - np.percentile(rms, 5))
        
        # Calculate breath control score
        breath_control = self._calculate_breath_control(y_mono, sr)
        
        # Calculate pitch accuracy
        pitch_accuracy = self._calculate_pitch_accuracy(valid_pitches, valid_magnitudes)
        
        # Calculate resonance score
        resonance_score = self._calculate_resonance(y_mono, sr)
        
        # Calculate articulation score
        onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
        onset_density = float(len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / (len(y_mono)/sr))
        articulation_score = self._rate_articulation(onset_density, mfcc_stds)
        
        # Calculate emotional expressivity
        emotional_expressivity = self._calculate_expressivity(mfcc_means, mfcc_stds, dynamic_range)
        
        return {
            "pitch_accuracy": float(pitch_accuracy),
            "min_pitch_hz": float(min_pitch),
            "max_pitch_hz": float(max_pitch),
            "vocal_range_semitones": float(vocal_range_semitones),
            "breath_control": float(breath_control),
            "vibrato_rate": float(vibrato_rate),
            "vibrato_extent": float(vibrato_extent),
            "vibrato_quality": float(self._rate_vibrato(vibrato_rate, vibrato_extent)),
            "resonance_score": float(resonance_score),
            "dynamic_range": float(dynamic_range),
            "articulation_score": float(articulation_score),
            "emotional_expressivity": float(emotional_expressivity),
            "spectral_centroid": float(spectral_centroid),
            "spectral_bandwidth": float(spectral_bandwidth),
            "spectral_contrast": spectral_contrast
        }

    def _detect_register_transitions(self, pitches, confidence, spectral_centroid):
        """Return default register transition points (for simplicity)."""
        return {
            "chest_to_mix_hz": 293.66,
            "chest_to_mix_note": "D4",
            "mix_to_head_hz": 659.26,
            "mix_to_head_note": "E5",
            "head_to_whistle_hz": 1046.50,
            "head_to_whistle_note": "C6"
        }

    def _hz_to_note(self, freq):
        """Convert a frequency in Hz to a musical note."""
        if freq < 20:
            return "N/A"
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        semitones = 12 * math.log2(freq/440)
        absolute_semitone = round(semitones) + 57
        octave = absolute_semitone // 12
        note_index = absolute_semitone % 12
        return f"{notes[note_index]}{octave}"

    def _analyze_vibrato(self, pitches, frame_length=2048, hop_length=512):
        """Analyze vibrato using a simple autocorrelation method."""
        if len(pitches) < 50:
            return 0, 0
        cents = 1200 * np.log2(pitches / np.mean(pitches))
        smooth_cents = np.convolve(cents, np.ones(5)/5, mode='valid')
        autocorr = np.correlate(smooth_cents, smooth_cents, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        peaks = librosa.util.peak_pick(autocorr, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.5, wait=10)
        if len(peaks) < 2:
            return 0, 0
        first_peak = peaks[0] if peaks[0] > 5 else peaks[1]
        if first_peak == 0:
            return 0, 0
        vibrato_rate = 44100 / (hop_length * first_peak)
        vibrato_extent = np.std(smooth_cents) * 2 if 4 < vibrato_rate < 8 else 0
        return vibrato_rate, vibrato_extent

    def _estimate_breathiness(self, y, sr):
        """Estimate breathiness using spectral flatness instead of harmonic ratio."""
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        breathiness = np.mean(spectral_flatness) * 10
        return min(10, breathiness)

    def _calculate_pitch_accuracy(self, pitches, confidence):
        """Calculate pitch accuracy based on stability and confidence."""
        if len(pitches) < 50:
            return 0
        pitch_diff = np.abs(np.diff(pitches))
        stable = pitch_diff < 10
        stability = np.mean(stable) * 10
        mean_conf = np.mean(confidence[confidence > 0.5]) if np.any(confidence > 0.5) else 0
        return (stability * 0.7 + mean_conf * 3) / 1.0

    def _calculate_breath_control(self, y, sr):
        """Estimate breath control from phrase length variability."""
        rms = librosa.feature.rms(y=y)[0]
        silent = rms < np.mean(rms) * 0.3
        phrases = []
        current = 0
        for v in silent:
            if not v:
                current += 1
            elif current > 0:
                phrases.append(current)
                current = 0
        if current > 0:
            phrases.append(current)
        if not phrases:
            return 0
        hop = 512
        phrase_secs = [p * hop / sr for p in phrases]
        mean_phrase = np.mean(phrase_secs)
        max_phrase = np.max(phrase_secs)
        stability = 1 - (np.std(phrase_secs)/(mean_phrase+1e-5))
        score = (mean_phrase * 2 + max_phrase + stability * 5) / 4
        return min(10, score)

    def _calculate_resonance(self, y, sr):
        """Estimate resonance using MFCC clarity and spectral properties."""
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        clarity = 1 - (np.std(mfccs[0])/(np.mean(mfccs[0])+1e-5)) if np.mean(mfccs[0]) != 0 else 0
        spec = np.abs(librosa.stft(y))
        mean_spec = np.mean(spec, axis=1)
        freqs = librosa.fft_frequencies(sr=sr)
        idx1 = np.argmin(np.abs(freqs-2000))
        idx2 = np.argmin(np.abs(freqs-4000))
        singers_formant = np.mean(mean_spec[idx1:idx2])/(np.mean(mean_spec[:idx1])+1e-5)
        resonance = (clarity * 3 + singers_formant * 5) / 9
        return min(10, resonance * 10)

    def _rate_articulation(self, onset_density, mfcc_stds):
        """Rate articulation based on onset density and MFCC variation."""
        onset_score = min(1, onset_density/3)*10
        mfcc_var = np.mean(mfcc_stds[1:5]) if len(mfcc_stds)>4 else np.mean(mfcc_stds)
        mfcc_score = min(1, mfcc_var/15)*10
        return (onset_score*0.7 + mfcc_score*0.3)

    def _rate_vibrato(self, rate, extent):
        """Rate vibrato quality based on rate and extent."""
        if rate == 0 or extent == 0:
            return 0
        rate_score = 10 - min(5, abs(rate-5.5))
        extent_score = 10 - min(5, abs(extent-40)/8)
        return (rate_score*0.6 + extent_score*0.4)

    def _calculate_expressivity(self, mfcc_means, mfcc_stds, dynamic_range):
        """Calculate emotional expressivity from dynamic range and timbre variation."""
        dyn_score = min(10, dynamic_range*20)
        timbre = np.mean(mfcc_stds)
        timbre_score = min(10, timbre*2.5)
        return (dyn_score*0.6 + timbre_score*0.4)

    def _estimate_musical_key(self, vocals, sr):
        """Estimate the musical key of the vocal performance."""
        try:
            # Convert to mono if necessary
            y_mono = np.mean(vocals, axis=0) if len(vocals.shape) > 1 else vocals
            
            # Extract chroma features
            chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr)
            
            # Average chroma over time
            chroma_avg = np.mean(chroma, axis=1)
            
            # Define key names
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            modes = ['Major', 'Minor']
            
            # Simple template matching for major and minor keys
            major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
            minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
            
            max_correlation = -1
            best_key = 'C'
            best_mode = 'Major'
            
            for i in range(12):  # For each possible key
                # Rotate templates to match each key
                major_rotated = np.roll(major_template, i)
                minor_rotated = np.roll(minor_template, i)
                
                # Calculate correlation
                major_corr = np.correlate(chroma_avg, major_rotated)
                minor_corr = np.correlate(chroma_avg, minor_rotated)
                
                if major_corr > max_correlation:
                    max_correlation = major_corr
                    best_key = keys[i]
                    best_mode = 'Major'
                
                if minor_corr > max_correlation:
                    max_correlation = minor_corr
                    best_key = keys[i]
                    best_mode = 'Minor'
            
            return f"{best_key} {best_mode}"
        except Exception as e:
            print(f"Error estimating key: {str(e)}")
            return "Unknown"

# Usage example (command-line):
if __name__ == "__main__":
    import sys
    analyzer = VocalAnalyzer()
    if len(sys.argv) > 1:
        result = analyzer.analyze_file(sys.argv[1])
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python vocal_analyzer.py path/to/audio.mp3")