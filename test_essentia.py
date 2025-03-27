import essentia
import essentia.standard as es
import numpy as np

# Print essentia version
print(f"Essentia version: {essentia.__version__}")

# Create a simple sine wave
sampleRate = 44100
duration = 1.0  # seconds
t = np.linspace(0, duration, int(sampleRate * duration))
audio = np.sin(2 * np.pi * 440 * t)

# Test some basic algorithms
print("\nTesting basic algorithms:")
print("1. RMS energy:", es.RMS()(audio))
print("2. Zero crossing rate:", es.ZeroCrossingRate()(audio))
print("3. Spectral centroid:", es.SpectralCentroid()(audio))

# Create a simple audio loader
loader = es.MonoLoader()

# Print available algorithms
print("\nAvailable algorithms:")
for algorithm in es.available_algorithms():
    print(f"- {algorithm}") 