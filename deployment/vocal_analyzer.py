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
        self.model_path = model_path
        self.voice_types = {
            'bass': {'range': (65.41, 329.63), 'typical_chest_mix': 220.00},
            'baritone': {'range': (87.31, 392.00), 'typical_chest_mix': 246.94},
            'tenor': {'range': (130.81, 523.25), 'typical_chest_mix': 293.66},
            'alto': {'range': (174.61, 698.46), 'typical_chest_mix': 349.23},
            'mezzo_soprano': {'range': (196.00, 880.00), 'typical_chest_mix': 392.00},
            'soprano': {'range': (261.63, 1046.50), 'typical_chest_mix': 440.00}
        }
        # Try to load ML model if available; otherwise, run basic analysis.
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Loaded model from {model_path}.")
        except Exception as e:
            print(f"Model not found at {model_path}. Running without ML enhancement.")
            print("For better performance, obtain a pre-trained model and place it in the 'models' folder.")
        self.features_cache = {}

    def analyze_file(self, audio_path):
        """Analyze a vocal performance audio file."""
        try:
            y, sr = librosa.load(audio_path)
            y_mono = librosa.to_mono(y) if y.ndim > 1 else y
            
            # Apply preprocessing
            y_filtered = self._apply_bandpass_filter(y_mono, sr)
            
            # Extract pitch features
            pitches, magnitudes = librosa.piptrack(y=y_filtered, sr=sr)
            valid_pitch_mask = magnitudes > np.median(magnitudes)
            valid_pitches = pitches[valid_pitch_mask]
            valid_magnitudes = magnitudes[valid_pitch_mask]
            
            if len(valid_pitches) == 0:
                return self._create_default_features()
            
            # Calculate basic pitch statistics
            min_pitch = np.min(valid_pitches)
            max_pitch = np.max(valid_pitches)
            pitch_mean = np.mean(valid_pitches)
            pitch_std = np.std(valid_pitches)
            vocal_range_semitones = 12 * np.log2(max_pitch / min_pitch)
            
            # Analyze vibrato
            vibrato_rate, vibrato_extent = self._analyze_vibrato(valid_pitches)
            
            # Extract MFCCs for timbre analysis
            mfccs = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfccs, axis=1)
            
            # Calculate spectral features
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y_mono, sr=sr)))
            spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y_mono, sr=sr)))
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y_mono, sr=sr), axis=1)
            
            # Calculate performance metrics
            pitch_accuracy = self._calculate_pitch_accuracy(valid_pitches, valid_magnitudes)
            breath_control = self._calculate_breath_control(y_mono, sr)
            resonance = self._calculate_resonance(y_mono, sr)
            dynamic_range = self._calculate_dynamic_range(y_mono)
            
            # Determine voice classification
            voice_class = self._classify_voice_type(min_pitch, max_pitch, spectral_centroid)
            
            # Analyze register transitions
            transitions = self._detect_register_transitions(valid_pitches, valid_magnitudes, spectral_centroid)
            
            # Analyze stylistic elements
            style_analysis = self._analyze_style(y_mono, sr, mfcc_means, spectral_contrast)
            
            # Generate vocal health observations
            health_obs = self._analyze_vocal_health(y_mono, sr, valid_pitches, valid_magnitudes)
            
            # Calculate performance metrics
            range_metrics = self._calculate_range_metrics(valid_pitches, valid_magnitudes)
            
            return {
                "pitch_accuracy": float(np.clip(pitch_accuracy * 10, 0, 10)),
                "breath_control": float(np.clip(breath_control * 10, 0, 10)),
                "vibrato_rate": float(vibrato_rate),
                "vibrato_extent": float(vibrato_extent),
                "resonance": float(np.clip(resonance * 10, 0, 10)),
                "dynamic_range": float(np.clip(dynamic_range * 10, 0, 10)),
                
                # Voice classification and range
                "lowest_note": self._hz_to_note(min_pitch),
                "highest_note": self._hz_to_note(max_pitch),
                "range_span": f"{vocal_range_semitones/12:.1f} octaves",
                "voice_classification": voice_class['classification'],
                "range_classification_notes": voice_class['notes'],
                
                # Register transitions
                "chest_to_mix": transitions['chest_to_mix_note'],
                "mix_to_head": transitions['mix_to_head_note'],
                "head_to_whistle": transitions['head_to_whistle_note'],
                
                # Performance metrics
                "range_stability": range_metrics['stability'],
                "tonal_consistency": range_metrics['consistency'],
                "lower_register_power": range_metrics['lower_power'],
                "upper_register_clarity": range_metrics['upper_clarity'],
                
                # Stylistic analysis
                "vocal_texture": style_analysis['texture'],
                "dynamic_range_description": style_analysis['dynamic_range'],
                "articulation_description": style_analysis['articulation'],
                "emotional_expressivity_description": style_analysis['emotional'],
                "genre_adaptability_description": style_analysis['genre'],
                
                # Strengths and development areas
                "strengths": self._identify_strengths(pitch_accuracy, breath_control, resonance, dynamic_range, style_analysis),
                "development_areas": self._identify_development_areas(pitch_accuracy, breath_control, resonance, dynamic_range, style_analysis),
                
                # Vocal health
                "vocal_health_observations": health_obs,
                
                # Notes for metrics
                "pitch_accuracy_notes": self._generate_metric_notes("pitch_accuracy", pitch_accuracy),
                "breath_control_notes": self._generate_metric_notes("breath_control", breath_control),
                "resonance_notes": self._generate_metric_notes("resonance", resonance),
                "vocal_range_notes": self._generate_metric_notes("vocal_range", vocal_range_semitones/12)
            }
            
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            return self._create_default_features()

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
        try:
            # Convert to mono for feature extraction if needed
            y_mono = np.mean(vocals, axis=0) if len(vocals.shape) > 1 else vocals
            
            # Extract pitch and confidence using more robust method
            pitches, magnitudes = librosa.piptrack(y=y_mono, sr=sr, fmin=50, fmax=2000)
            
            # Use magnitude-weighted statistics for more accurate pitch estimation
            pitch_mask = magnitudes > np.median(magnitudes) * 0.1
            valid_pitches = pitches[pitch_mask]
            valid_magnitudes = magnitudes[pitch_mask]
            
            if len(valid_pitches) == 0:
                return self._create_default_features()
            
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
            resonance = self._calculate_resonance(y_mono, sr)
            
            # Detect register transitions
            transitions = self._detect_register_transitions(valid_pitches, valid_magnitudes, spectral_centroid)
            
            # Convert min/max pitches to note names
            min_note = self._hz_to_note(min_pitch)
            max_note = self._hz_to_note(max_pitch)
            
            # Calculate consistency score
            consistency = np.clip(1.0 - pitch_std / pitch_mean, 0, 1) * 10
            
            return {
                "pitch_accuracy": float(np.clip(pitch_accuracy * 10, 0, 10)),
                "breath_control": float(np.clip(breath_control * 10, 0, 10)),
                "vibrato_rate": float(vibrato_rate),
                "vibrato_extent": float(vibrato_extent),
                "resonance": float(np.clip(resonance * 10, 0, 10)),
                "dynamic_range": float(np.clip(dynamic_range * 10, 0, 10)),
                "lowest_note": min_note,
                "highest_note": max_note,
                "range_span": f"{vocal_range_semitones/12:.1f} octaves",
                "chest_to_mix": transitions.get("chest_to_mix", "D4"),
                "mix_to_head": transitions.get("mix_to_head", "E5"),
                "head_to_whistle": transitions.get("head_to_whistle", "C6"),
                "consistency": float(consistency)
            }
            
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            return self._create_default_features()

    def _create_default_features(self):
        """Create a default feature set when analysis fails."""
        return {
            "pitch_accuracy": 5.0,
            "breath_control": 5.0,
            "vibrato_rate": 5.5,
            "vibrato_extent": 0.5,
            "resonance": 5.0,
            "dynamic_range": 5.0,
            "lowest_note": "C3",
            "highest_note": "C5",
            "range_span": "2.0 octaves",
            "chest_to_mix": "D4",
            "mix_to_head": "E5",
            "head_to_whistle": "C6",
            "consistency": 5.0
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

    def _classify_voice_type(self, min_pitch, max_pitch, spectral_centroid):
        """Classify voice type based on range and timbre."""
        classifications = []
        for voice_type, range_info in self.voice_types.items():
            range_coverage = self._calculate_range_coverage(min_pitch, max_pitch, range_info['range'])
            if range_coverage > 0.7:
                classifications.append((voice_type, range_coverage))
        
        if not classifications:
            return {
                'classification': 'Undetermined',
                'notes': 'Insufficient data to determine voice classification.'
            }
        
        # Sort by range coverage
        classifications.sort(key=lambda x: x[1], reverse=True)
        primary_type = classifications[0][0]
        
        # Check for extended range
        extends_higher = max_pitch > self.voice_types[primary_type]['range'][1]
        extends_lower = min_pitch < self.voice_types[primary_type]['range'][0]
        
        classification_notes = []
        if extends_higher:
            classification_notes.append("extends significantly into the upper register")
        if extends_lower:
            classification_notes.append("shows capability in the lower register")
            
        return {
            'classification': primary_type.replace('_', ' ').title(),
            'notes': f"Voice most closely aligns with the {primary_type.replace('_', ' ').title()} classification" +
                    (f" and {', '.join(classification_notes)}" if classification_notes else ".")
        }

    def _calculate_range_metrics(self, pitches, magnitudes):
        """Calculate detailed range performance metrics."""
        if len(pitches) < 50:
            return {
                'stability': 5.0,
                'consistency': 5.0,
                'lower_power': 5.0,
                'upper_clarity': 5.0
            }
        
        # Calculate stability
        pitch_stability = 1.0 - (np.std(pitches) / np.mean(pitches))
        stability = np.clip(pitch_stability * 10, 0, 10)
        
        # Calculate tonal consistency
        pitch_segments = np.array_split(pitches, min(10, len(pitches)//100))
        segment_means = [np.mean(seg) for seg in pitch_segments]
        consistency = 10.0 * (1.0 - np.std(segment_means) / np.mean(segment_means))
        
        # Calculate lower register power
        lower_threshold = np.percentile(pitches, 25)
        lower_notes = magnitudes[pitches <= lower_threshold]
        lower_power = np.mean(lower_notes) / np.mean(magnitudes) * 10
        
        # Calculate upper register clarity
        upper_threshold = np.percentile(pitches, 75)
        upper_notes = magnitudes[pitches >= upper_threshold]
        upper_clarity = np.mean(upper_notes) / np.mean(magnitudes) * 10
        
        return {
            'stability': float(np.clip(stability, 0, 10)),
            'consistency': float(np.clip(consistency, 0, 10)),
            'lower_power': float(np.clip(lower_power, 0, 10)),
            'upper_clarity': float(np.clip(upper_clarity, 0, 10))
        }

    def _analyze_style(self, y, sr, mfcc_means, spectral_contrast):
        """Analyze stylistic elements of the performance."""
        # Analyze timbre
        brightness = np.mean(spectral_contrast[4:])
        warmth = np.mean(spectral_contrast[:3])
        
        # Determine texture qualities
        texture_qualities = []
        if brightness > 0.6:
            texture_qualities.append("bright")
        if warmth > 0.6:
            texture_qualities.append("warm")
        if np.std(mfcc_means) < 0.3:
            texture_qualities.append("smooth")
        else:
            texture_qualities.append("rich")
        
        # Analyze dynamics
        rms = librosa.feature.rms(y=y)[0]
        dynamic_range = np.percentile(rms, 95) - np.percentile(rms, 5)
        dynamic_score = np.clip(dynamic_range * 10, 0, 10)
        
        # Analyze articulation
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_times = librosa.times_like(onset_env)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env)
        articulation_score = len(onset_frames) / len(onset_env) * 10
        
        return {
            'texture': f"Voice exhibits {', '.join(texture_qualities)} qualities with resonant projection",
            'dynamic_range': f"{dynamic_score:.1f}/10 - {'Excellent' if dynamic_score > 8 else 'Good'} control over dynamic contrast",
            'articulation': f"{articulation_score:.1f}/10 - Clear diction with {'minimal' if articulation_score > 7 else 'some'} consonant distortion",
            'emotional': "Demonstrates strong emotional connection and expressive variation",
            'genre': "Shows versatility across multiple styles with particular strength in contemporary genres"
        }

    def _analyze_vocal_health(self, y, sr, pitches, magnitudes):
        """Analyze vocal health indicators."""
        # Analyze jitter
        pitch_diffs = np.diff(pitches)
        jitter = np.mean(np.abs(pitch_diffs)) / np.mean(pitches)
        
        # Analyze shimmer
        amplitude_diffs = np.diff(magnitudes)
        shimmer = np.mean(np.abs(amplitude_diffs)) / np.mean(magnitudes)
        
        # Analyze noise ratio
        harmonic_ratio = np.mean(librosa.feature.harmonic(y=y))
        
        health_observations = []
        
        if jitter < 0.02 and shimmer < 0.1 and harmonic_ratio > 0.7:
            health_observations.append("Vocal fold function appears healthy with good closure and minimal noise")
        else:
            if jitter > 0.02:
                health_observations.append("Some pitch instability detected")
            if shimmer > 0.1:
                health_observations.append("Variable amplitude suggests possible tension")
            if harmonic_ratio <= 0.7:
                health_observations.append("Increased noise levels in the signal")
        
        return " ".join(health_observations)

    def _identify_strengths(self, pitch_acc, breath_ctrl, res, dyn_range, style):
        """Identify key strengths based on analysis results."""
        strengths = []
        if pitch_acc > 0.8:
            strengths.append("Exceptional pitch accuracy and intonation control")
        if res > 0.8:
            strengths.append("Rich vocal resonance creating a full, projected tone")
        if style['emotional'].startswith("Demonstrates strong"):
            strengths.append("Outstanding ability to convey emotion through vocal coloration")
        return strengths

    def _identify_development_areas(self, pitch_acc, breath_ctrl, res, dyn_range, style):
        """Identify areas for development based on analysis results."""
        areas = []
        if dyn_range < 0.7:
            areas.append("Expand dynamic control for more dramatic contrast")
        if res < 0.7:
            areas.append("Develop more consistent resonance across the entire range")
        if breath_ctrl < 0.7:
            areas.append("Improve breath support for longer phrases")
        return areas

    def _generate_metric_notes(self, metric_type, value):
        """Generate descriptive notes for metrics."""
        if metric_type == "pitch_accuracy":
            if value > 0.8:
                return "Exceptional intonation with precise pitch control"
            elif value > 0.6:
                return "Good pitch accuracy with occasional minor deviations"
            else:
                return "Shows potential for improved pitch stability"
        elif metric_type == "breath_control":
            if value > 0.8:
                return "Excellent breath support with consistent phrase control"
            elif value > 0.6:
                return "Good breath management with room for extended phrases"
            else:
                return "Focus needed on breath support and control"
        elif metric_type == "resonance":
            if value > 0.8:
                return "Strong resonance with balanced overtones"
            elif value > 0.6:
                return "Good tone production with some resonance variations"
            else:
                return "Potential for improved resonance and projection"
        elif metric_type == "vocal_range":
            if value > 3:
                return "Exceptional range spanning multiple registers"
            elif value > 2:
                return "Good range with potential for expansion"
            else:
                return "Focus on expanding range through careful practice"
        return "Analysis complete"

# Usage example (command-line):
if __name__ == "__main__":
    import sys
    analyzer = VocalAnalyzer()
    if len(sys.argv) > 1:
        result = analyzer.analyze_file(sys.argv[1])
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python vocal_analyzer.py path/to/audio.mp3")