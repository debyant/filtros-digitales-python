import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz

# ==========================================
# 1. DEFINICIÓN DE LA SEÑAL DE ENTRADA
# ==========================================
# Parámetros de muestreo
fs = 1000.0  # Frecuencia de muestreo (Hz)
t_final = 1.0  # Duración en segundos
t = np.linspace(0, t_final, int(fs * t_final), endpoint=False)

# Creación de una señal compuesta por 3 frecuencias: 5 Hz (baja), 50 Hz (media), 150 Hz (alta)
f1, f2, f3 = 5, 50, 150
seno_limpio = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t) + 0.2 * np.sin(2 * np.pi * f3 * t)

# Adición de Ruido Blanco Gaussia
np.random.seed(42)  # Semilla para reproducibilidad
ruido = np.random.normal(0, 0.5, t.shape)
senal_entrada = seno_limpio + ruido

# Función auxiliar para calcular el espectro en frecuencia (FFT)
def calcular_fft(signal, fs):
    N = len(signal)
    fft_val = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    # Tomamos la mitad positiva del espectro
    pos_mask = freqs >= 0
    return freqs[pos_mask], np.abs(fft_val[pos_mask]) / (N / 2)

# ==========================================
# 2. DISEÑO Y APLICACIÓN DE FILTROS IIR (Butterworth)
# ==========================================
orden = 4  # Orden del filtro

# --- A) Filtro Pasa Bajos (Low-Pass) ---
fc_low = 15.0  # Corte a 15 Hz (deja pasar la señal de 5 Hz)
b_lp, a_lp = butter(orden, fc_low, btype='low', fs=fs)
filtrada_lp = filtfilt(b_lp, a_lp, senal_entrada)

# --- B) Filtro Pasa Altos (High-Pass) ---
fc_high = 100.0  # Corte a 100 Hz (deja pasar la señal de 150 Hz)
b_hp, a_hp = butter(orden, fc_high, btype='high', fs=fs)
filtrada_hp = filtfilt(b_hp, a_hp, senal_entrada)

# --- C) Filtro Pasa Bandas (Band-Pass) ---
fc_band = [30.0, 70.0]  # Banda entre 30 Hz y 70 Hz (deja pasar la señal de 50 Hz)
b_bp, a_bp = butter(orden, fc_band, btype='bandpass', fs=fs)
filtrada_bp = filtfilt(b_bp, a_bp, senal_entrada)

# ==========================================
# 3. VISUALIZACIÓN DE LOS RESULTADOS
# ==========================================

# A) Gráficas en el Dominio del Tiempo
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(t, senal_entrada, color='gray', alpha=0.7, label='Señal Ruidosa de Entrada')
plt.title('Análisis en el Dominio del Tiempo')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, filtrada_lp, color='blue', label='Filtro Pasa Bajos (fc = 15 Hz)')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, filtrada_hp, color='red', label='Filtro Pasa Altos (fc = 100 Hz)')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, filtrada_bp, color='green', label='Filtro Pasa Bandas (30-70 Hz)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

# B) Gráficas en el Dominio de la Frecuencia (FFT)
freqs, fft_entrada = calcular_fft(senal_entrada, fs)
_, fft_lp = calcular_fft(filtrada_lp, fs)
_, fft_hp = calcular_fft(filtrada_hp, fs)
_, fft_bp = calcular_fft(filtrada_bp, fs)

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(freqs, fft_entrada, color='gray')
plt.title('Análisis en el Dominio de la Frecuencia (Espectro FFT)')
plt.ylabel('Magnitud')
plt.xlim(0, 200)
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(freqs, fft_lp, color='blue')
plt.ylabel('Magnitud (LPF)')
plt.xlim(0, 200)
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(freqs, fft_hp, color='red')
plt.ylabel('Magnitud (HPF)')
plt.xlim(0, 200)
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(freqs, fft_bp, color='green')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud (BPF)')
plt.xlim(0, 200)
plt.grid(True)

plt.tight_layout()
plt.show()

# C) Respuesta en Frecuencia de los Filtros (Diagramas de Bode de Magnitud)
w_lp, h_lp = freqz(b_lp, a_lp, fs=fs)
w_hp, h_hp = freqz(b_hp, a_hp, fs=fs)
w_bp, h_bp = freqz(b_bp, a_bp, fs=fs)

plt.figure(figsize=(10, 5))
plt.plot(w_lp, 20 * np.log10(abs(h_lp)), label='Pasa Bajos (fc=15Hz)', color='blue')
plt.plot(w_hp, 20 * np.log10(abs(h_hp)), label='Pasa Altos (fc=100Hz)', color='red')
plt.plot(w_bp, 20 * np.log10(abs(h_bp)), label='Pasa Bandas (30-70Hz)', color='green')
plt.title('Respuesta en Frecuencia de los Filtros Diseñados')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Ganancia [dB]')
plt.ylim(-60, 5)
plt.xlim(0, 200)
plt.axhline(-3, color='black', linestyle='--', alpha=0.7, label='Corte a -3dB')
plt.grid(True)
plt.legend()
plt.show()
