import whisper
import sounddevice as sd
import scipy.io.wavfile
import time
import os
from datetime import datetime

# Configuración
DURACION_SEGUNDOS = 5
TIEMPO_ENTRE_CICLOS = 1

modelo = whisper.load_model("base")

def cargar_frases(ruta="frases.txt"):
    with open(ruta, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]

frases_objetivo = cargar_frases()

def grabar_audio(duracion=DURACION_SEGUNDOS, samplerate=16000):
    print("\n🎤 Grabando...")
    audio = sd.rec(int(duracion * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    return audio, samplerate

def transcribir(audio, samplerate):
    ruta_wav = "grabacion.wav"
    scipy.io.wavfile.write(ruta_wav, samplerate, audio)
    time.sleep(0.5)
    if not os.path.exists(ruta_wav):
        raise FileNotFoundError(f"No se encontró el archivo {ruta_wav}")
    resultado = modelo.transcribe(os.path.abspath(ruta_wav), language="es")
    return resultado["text"].lower()

def detectar_frase_en_texto(texto, frases):
    for frase in frases:
        if frase in texto:
            return frase
    return None

def guardar_log(frase_detectada):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = f"[{timestamp}] Frase detectada: {frase_detectada}\n"
    with open("detecciones.log", "a", encoding="utf-8") as log:
        log.write(linea)

def vigilancia():
    print("🟢 Iniciando vigilancia de voz indefinida... (Ctrl+C para salir)")

    while True:
        try:
            audio, samplerate = grabar_audio()
            texto = transcribir(audio, samplerate)
            print("📝 Transcripción:", texto)

            frase_detectada = detectar_frase_en_texto(texto, frases_objetivo)
            if frase_detectada:
                print(f"⚠️ Frase detectada: '{frase_detectada}'")
                guardar_log(frase_detectada)

            time.sleep(TIEMPO_ENTRE_CICLOS)

        except KeyboardInterrupt:
            print("\n🛑 Vigilancia detenida por el usuario.")
            break
        except Exception as e:
            print("❌ Error:", e)
            time.sleep(2)

if __name__ == "__main__":
    vigilancia()
