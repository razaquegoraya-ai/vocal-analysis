import librosa
import numpy as np
import essentia.standard as es
from scipy.stats import pearsonr
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
        self.pitch_processor = es.PredominantPitchMelodia()
        self.features_cache = {}

    def analyze_file(self, file_path):
        """Analyze a single audio file and extract vocal features."""
        try:
            y, sr = librosa.load(file_path, sr=44100)
        except Exception as e:
            return {"error": f"Failed to load audio file: {str(e)}"}
        vocals = self._extract_vocals(y, sr)
        if len(vocals) == 0:
            return {"error": "No vocal content detected"}
        features = self._extract_features(vocals, sr)
        self.features_cache[Path(file_path).name] = features
        return features

    def _extract_vocals(self, y, sr):
        """Extract vocal segments using the VocalSeparator."""
        from vocal_separator import VocalSeparator
        try:
            separator = VocalSeparator(method='spleeter')
            vocals = separator.separate(y, sr)
            separator.cleanup()
            return vocals
        except Exception as e:
            print(f"Advanced vocal separation failed: {str(e)}. Falling back to basic separation.")
        y_harmonic, _ = librosa.effects.hpss(y)
        spec = np.abs(librosa.stft(y_harmonic))
        spec_sum = np.sum(spec, axis=0)
        threshold = np.mean(spec_sum) * 1.5
        mask = spec_sum > threshold
        segments = []
        start = None
        for i, m in enumerate(mask):
            if m and start is None:
                start = i
            elif not m and start is not None:
                end = i
                if end - start > sr // 2:
                    segments.append((start, end))
                start = None
        vocals = []
        for start, end in segments:
            start_time = librosa.frames_to_time(start, sr=sr)
            end_time = librosa.frames_to_time(end, sr=sr)
            segment = y[int(start_time * sr):int(end_time * sr)]
            vocals.append(segment)
        return np.concatenate(vocals) if vocals else np.array([])

    def _extract_features(self, vocals, sr):
        """Extract core vocal features and compute additional metrics."""
        pitches, confidence = librosa.piptrack(y=vocals, sr=sr)
        pitch_values = pitches[pitches > 0]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
        pitch_vals, pitch_conf = self.pitch_processor(vocals)
        valid_pitches = pitch_vals[pitch_conf > 0.8]
        if len(valid_pitches) > 0:
            min_pitch = np.min(valid_pitches[valid_pitches > 50])
            max_pitch = np.max(valid_pitches[valid_pitches < 2000])
            vocal_range_semitones = 12 * np.log2(max_pitch/min_pitch) if min_pitch > 0 else 0
        else:
            min_pitch = max_pitch = vocal_range_semitones = 0
        vibrato_rate, vibrato_extent = self._analyze_vibrato(valid_pitches)
        mfccs = librosa.feature.mfcc(y=vocals, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=vocals, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=vocals, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=vocals, sr=sr), axis=1)
        rms = librosa.feature.rms(y=vocals)[0]
        dynamic_range = np.percentile(rms, 95) - np.percentile(rms, 5)
        breathiness = self._estimate_breathiness(vocals, sr)
        onset_env = librosa.onset.onset_strength(y=vocals, sr=sr)
        onset_density = len(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)) / (len(vocals)/sr)
        pitch_accuracy = self._calculate_pitch_accuracy(valid_pitches, pitch_conf)
        breath_control = self._calculate_breath_control(vocals, sr)
        resonance_score = self._calculate_resonance(vocals, sr)
        articulation_score = self._rate_articulation(onset_density, mfcc_stds)
        emotional_expressivity = self._calculate_expressivity(mfcc_means, mfcc_stds, dynamic_range)
        register_transitions = self._detect_register_transitions(valid_pitches, pitch_conf, spectral_centroid)
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
            "pitch_confidence": float(np.mean(pitch_conf)) if len(pitch_conf) > 0 else 0,
            "spectral_centroid": float(spectral_centroid),
            "spectral_bandwidth": float(spectral_bandwidth),
            "spectral_contrast": spectral_contrast.tolist(),
            "register_transitions": register_transitions
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
        """Estimate breathiness using harmonic ratio (via Essentia)."""
        harmonic = es.HarmonicRatio()
        frame_size = 2048
        hop_size = 1024
        hnr_values = []
        for i in range(0, len(y)-frame_size, hop_size):
            frame = y[i:i+frame_size]
            if np.std(frame) > 0.01:
                try:
                    hnr = harmonic(frame)
                    hnr_values.append(hnr)
                except Exception:
                    continue
        if not hnr_values:
            return 0
        mean_hnr = np.mean(hnr_values)
        breathiness = 10 / (1 + mean_hnr)
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

# Usage example (command-line):
if __name__ == "__main__":
    import sys
    analyzer = VocalAnalyzer()
    if len(sys.argv) > 1:
        result = analyzer.analyze_file(sys.argv[1])
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python vocal_analyzer.py path/to/audio.mp3")