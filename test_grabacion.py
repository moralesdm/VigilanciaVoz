import sounddevice as sd
import scipy.io.wavfile

samplerate = 16000
duration = 5
filename = "grabacion.wav"

print("🎤 Grabando...")
audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
sd.wait()
scipy.io.wavfile.write(filename, samplerate, audio)
print(f"✅ Grabación guardada en {filename}")
