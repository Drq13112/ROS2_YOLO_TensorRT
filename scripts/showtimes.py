import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# Se espera que el archivo tenga 10 columnas, con las siguientes unidades:
#   0: timestamp                        (segundos)
#   1: buffer_copy_time_us              (microsegundos) -> se convertirá a ms
#   2: pre_processing_time_ms           (milisegundos)
#   3: inference_time_ms                (milisegundos)
#   4: post_processing_time_ms          (milisegundos)
#   5: total_processing_time_ms         (milisegundos)
#   6: batch_timestamp_diff_ms          (milisegundos)
#   7: avg_mask_generation_time_us      (microsegundos) -> se convertirá a ms
#   8: avg_message_creation_time_us     (microsegundos) -> se convertirá a ms
#   9: avg_publish_call_time_us         (microsegundos) -> se convertirá a ms
headers = [
    "timestamp",
    "buffer_copy_time_us",
    "pre_processing_time_ms",
    "inference_time_ms",
    "post_processing_time_ms",
    "total_processing_time_ms",
    "batch_timestamp_diff_ms",
    "avg_mask_generation_time_us",
    "avg_message_creation_time_us",
    "avg_publish_call_time_us"
]

# Factores de conversión: convertir microsegundos a milisegundos para:
#   buffer_copy, avg_mask_generation, avg_message_creation, avg_publish_call
conversion_factors = [
    1,          # timestamp (en segundos)
    1/1000.0,   # buffer_copy_time_us -> ms
    1,          # pre_processing_time_ms (ya en ms)
    1,          # inference_time_ms (ms)
    1,          # post_processing_time_ms (ms)
    1,          # total_processing_time_ms (ms)
    1,          # batch_timestamp_diff_ms (ms)
    1/1000.0,   # avg_mask_generation_time_us -> ms
    1/1000.0,   # avg_message_creation_time_us -> ms
    1/1000.0    # avg_publish_call_time_us -> ms
]

data_list = []
with open("frame_times.txt", "r") as f:
    reader = csv.reader(f, delimiter=",")
    line_num = 0
    for row in reader:
        line_num += 1
        row = [r.strip() for r in row if r.strip() != ""]
        if len(row) == 10:
            try:
                values = [float(x) for x in row]
                data_list.append(values)
            except ValueError:
                print(f"Línea {line_num}: Error de conversión. Se omite {row}")
        else:
            print(f"Línea {line_num}: Se omite porque tiene {len(row)} columnas en lugar de 10:\n {row}")

if not data_list:
    print("No se han encontrado datos válidos en el fichero.")
    exit()

data = np.array(data_list)

# Aplicar conversiones a las series (excepto el timestamp)
for i in range(1, data.shape[1]):
    data[:, i] *= conversion_factors[i]

# Mostrar estadísticas para cada columna
print("Estadísticas de tiempos por medida (media, máximo, mínimo, varianza):")
for i, col_name in enumerate(headers):
    col = data[:, i]
    mean_val = np.mean(col)
    max_val = np.max(col)
    min_val = np.min(col)
    var_val = np.var(col)
    print(f"{col_name:30s}: mean={mean_val:8.2f}, max={max_val:8.2f}, min={min_val:8.2f}, var={var_val:8.2f}")

#-----------------------------------------------------------
# Preparar el plot: cada serie se suaviza con un promedio móvil
#-----------------------------------------------------------
try:
    plt.style.use("seaborn-darkgrid")
except OSError:
    print("El estilo 'seaborn-darkgrid' no está disponible. Se usará 'seaborn'.")
    if "seaborn" in plt.style.available:
        plt.style.use("seaborn")
    else:
        plt.style.use("default")

timestamps = data[:, 0]  # Timestamp en segundos
num_series = data.shape[1] - 1  # Excluir el timestamp

def smooth_series(series, window=10):
    return pd.Series(series).rolling(window, min_periods=1, center=True).mean().values

colors = cm.get_cmap("tab20", num_series)

plt.figure(figsize=(14, 8))
# Graficar cada serie (columnas 1 a 9) en milisegundos
for i in range(1, data.shape[1]):
    smoothed = smooth_series(data[:, i], window=10)
    plt.plot(timestamps, smoothed,
             label=headers[i],
             color=colors(i-1),
             marker='o', markersize=3, linewidth=2.5, linestyle='-')

plt.xlabel("Timestamp (iters)", fontsize=18)
plt.ylabel("Computing time (ms)", fontsize=18)
plt.title("Tiempos medidos a lo largo del tiempo\n", fontsize=18)
plt.legend(loc="upper right", fontsize=14, ncol=3)
plt.grid(True)
plt.tight_layout()
plt.show()