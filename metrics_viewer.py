import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_data(csv_filepath):
    """Carga los datos del archivo CSV a un DataFrame de pandas."""
    try:
        df = pd.read_csv(csv_filepath)
        print(f"Datos cargados exitosamente desde {csv_filepath}")
        print(f"Columnas disponibles: {df.columns.tolist()}")
        # Convertir columnas relevantes a numérico, errores a NaN
        numeric_cols = [
            'packet_seq_num', 'msg_hdr_seq_nsec',
            't0_imgpub_sec', 't0_imgpub_nsec',
            't1_segnode_cb_sec', 't1_segnode_cb_nsec',
            't2_segnode_batchstart_sec', 't2_segnode_batchstart_nsec',
            't3_segnode_respub_sec', 't3_segnode_respub_nsec',
            't4_segsub_resrecv_sec', 't4_segsub_resrecv_nsec',
            'lat_imgpub_to_cb_ms', 'lat_cb_to_batch_start_ms',
            'lat_batch_start_to_res_pub_ms', 'lat_res_pub_to_res_recv_ms',
            'lat_segnode_cb_to_segsub_recv_ms', 'lat_total_e2e_ms',
            'offset_T0_front_vs_left_ms', 'offset_T0_right_vs_left_ms',
            'lost_pkts_since_last', 'total_lost_pkts_cam'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"Advertencia: La columna esperada '{col}' no se encontró en el CSV.")
        return df
    except FileNotFoundError:
        print(f"Error: El archivo {csv_filepath} no fue encontrado.")
        return None
    except Exception as e:
        print(f"Error cargando el archivo CSV: {e}")
        return None

def calculate_and_print_latency_stats(df):
    """Calcula y muestra estadísticas para las latencias especificadas."""
    if df is None or df.empty:
        print("No hay datos para calcular estadísticas de latencia.")
        return

    print("\n--- Estadísticas de Latencia (ms) ---")
    latency_intervals = {
        "T0->T1 (ImgPub a SegNode CB)": "lat_imgpub_to_cb_ms",
        "T1->T2 (SegNode CB a BatchStart)": "lat_cb_to_batch_start_ms",
        "T2->T3 (SegNode BatchStart a ResPub)": "lat_batch_start_to_res_pub_ms",
        "T3->T4 (SegNode ResPub a SegSub Recv)": "lat_res_pub_to_res_recv_ms",
        "T0->T4 (Total E2E)": "lat_total_e2e_ms"
    }

    for camera_id in df['camera_id'].unique():
        print(f"\nCámara: {camera_id}")
        df_camera = df[df['camera_id'] == camera_id].copy()
        
        for desc, col_name in latency_intervals.items():
            if col_name in df_camera.columns and not df_camera[col_name].isnull().all():
                latencies = df_camera[col_name].dropna()
                if not latencies.empty:
                    mean_lat = latencies.mean()
                    min_lat = latencies.min()
                    max_lat = latencies.max()
                    median_lat = latencies.median()
                    std_lat = latencies.std()
                    print(f"  {desc}:")
                    print(f"    Media: {mean_lat:.3f}, Mediana: {median_lat:.3f}, Mín: {min_lat:.3f}, Máx: {max_lat:.3f}, StdDev: {std_lat:.3f}")
                else:
                    print(f"  {desc}: No hay datos válidos.")
            else:
                print(f"  {desc}: Columna '{col_name}' no encontrada o todos los valores son NaN.")
    return latency_intervals # Devuelve para usar en análisis posterior

def calculate_and_print_loss_stats(df):
    """Calcula y muestra estadísticas de paquetes perdidos."""
    if df is None or df.empty:
        print("No hay datos para calcular estadísticas de pérdida.")
        return None

    print("\n--- Estadísticas de Paquetes Perdidos ---")
    loss_summary = []

    for camera_id in df['camera_id'].unique():
        df_camera = df[df['camera_id'] == camera_id].copy()
        if df_camera.empty or 'total_lost_pkts_cam' not in df_camera.columns or 'packet_seq_num' not in df_camera.columns:
            print(f"Cámara {camera_id}: Datos insuficientes para análisis de pérdidas.")
            continue

        # Usar el último valor registrado de total_lost_pkts_cam para esa cámara
        # Asegurarse de que la columna no tenga NaNs antes de tomar el último
        df_camera_loss_col = df_camera['total_lost_pkts_cam'].dropna()
        if df_camera_loss_col.empty:
            total_lost = 0
            print(f"Cámara {camera_id}: No hay datos válidos en 'total_lost_pkts_cam'. Asumiendo 0 pérdidas.")
        else:
            total_lost = df_camera_loss_col.iloc[-1]

        num_received = len(df_camera)
        
        # Para el porcentaje, el total de paquetes esperados es recibidos + perdidos
        total_expected_packets = num_received + total_lost
        
        percentage_lost = 0
        if total_expected_packets > 0:
            percentage_lost = (total_lost / total_expected_packets) * 100
        else:
            percentage_lost = 0 # Evitar división por cero si no se recibieron ni se perdieron paquetes (improbable)

        print(f"Cámara: {camera_id}")
        print(f"  Paquetes recibidos (medidos por seg_sub): {num_received}")
        print(f"  Paquetes perdidos (según 'total_lost_pkts_cam'): {total_lost:.0f}")
        print(f"  Porcentaje de pérdida: {percentage_lost:.2f}% (respecto a recibidos + perdidos)")
        loss_summary.append({'camera_id': camera_id, 'total_lost': total_lost, 'percentage_lost': percentage_lost, 'num_received': num_received})
        
    return pd.DataFrame(loss_summary)


def analyze_and_print_t0_offsets(df):
    """Analiza y muestra estadísticas de los desfases T0."""
    if df is None or df.empty:
        print("No hay datos para analizar desfases T0.")
        return

    print("\n--- Análisis de Desfase T0 (ms) ---")
    offset_cols = {
        "Offset T0 Front vs Left": "offset_T0_front_vs_left_ms",
        "Offset T0 Right vs Left": "offset_T0_right_vs_left_ms"
    }
    for desc, col_name in offset_cols.items():
        if col_name in df.columns and not df[col_name].isnull().all():
            offsets = df[col_name].dropna()
            if not offsets.empty:
                mean_offset = offsets.mean()
                min_offset = offsets.min()
                max_offset = offsets.max()
                median_offset = offsets.median()
                std_offset = offsets.std()
                print(f"  {desc}:")
                print(f"    Media: {mean_offset:.3f}, Mediana: {median_offset:.3f}, Mín: {min_offset:.3f}, Máx: {max_offset:.3f}, StdDev: {std_offset:.3f}")
            else:
                print(f"  {desc}: No hay datos válidos.")
        else:
            print(f"  {desc}: Columna '{col_name}' no encontrada o todos los valores son NaN.")

def evaluate_processing_performance(df, latency_intervals, loss_stats_df):
    """Evalúa si los problemas se deben a retrasos o saltos de paquetes."""
    if df is None or df.empty or loss_stats_df is None or loss_stats_df.empty:
        print("\nNo hay suficientes datos para la evaluación de rendimiento.")
        return

    print("\n--- Evaluación de Rendimiento: Retrasos vs. Saltos de Paquetes ---")
    
    camera_ids = df['camera_id'].unique()
    if len(camera_ids) < 2:
        print("Se necesita información de al menos dos cámaras para una comparación detallada de retrasos relativos.")
    
    # 1. Análisis de Retrasos (comparando latencias medias entre cámaras)
    print("\nAnálisis de Retrasos (comparación de latencias medias por etapa):")
    stage_latency_data = {} # camera_id -> {stage_desc: mean_latency}

    for camera_id in camera_ids:
        stage_latency_data[camera_id] = {}
        df_camera = df[df['camera_id'] == camera_id]
        for desc, col_name in latency_intervals.items():
            if col_name in df_camera.columns and not df_camera[col_name].isnull().all():
                latencies = df_camera[col_name].dropna()
                if not latencies.empty:
                    stage_latency_data[camera_id][desc] = latencies.mean()
                else:
                    stage_latency_data[camera_id][desc] = np.nan
            else:
                 stage_latency_data[camera_id][desc] = np.nan


    # Imprimir comparación de retrasos
    for desc in latency_intervals.keys():
        print(f"  Etapa '{desc}':")
        means = {cam: stage_latency_data[cam].get(desc, np.nan) for cam in camera_ids}
        valid_means = {cam: mean for cam, mean in means.items() if not np.isnan(mean)}
        
        if not valid_means:
            print("    No hay datos de latencia media para esta etapa en ninguna cámara.")
            continue

        for cam, mean_val in valid_means.items():
            print(f"    - {cam}: {mean_val:.3f} ms")
        
        if len(valid_means) > 1:
            sorted_means = sorted(valid_means.items(), key=lambda item: item[1])
            if sorted_means[-1][1] > sorted_means[0][1] * 1.2: # Si la diferencia es > 20%
                print(f"    * Observación: La cámara '{sorted_means[-1][0]}' es notablemente más lenta en esta etapa que '{sorted_means[0][0]}'.")

    # 2. Análisis de Saltos de Paquetes (usando loss_stats_df)
    print("\nAnálisis de Saltos de Paquetes (basado en porcentaje de pérdida total):")
    if not loss_stats_df.empty:
        sorted_loss = loss_stats_df.sort_values(by='percentage_lost', ascending=False)
        for _, row in sorted_loss.iterrows():
            print(f"  Cámara {row['camera_id']}: {row['percentage_lost']:.2f}% de paquetes perdidos ({row['total_lost']:.0f} de {row['num_received'] + row['total_lost']:.0f} esperados).")
        
        if len(sorted_loss) > 1 and sorted_loss.iloc[0]['percentage_lost'] > 0:
            if sorted_loss.iloc[0]['percentage_lost'] > sorted_loss.iloc[-1]['percentage_lost'] + 5: # Diferencia de al menos 5%
                 print(f"    * Observación: El camino de la cámara '{sorted_loss.iloc[0]['camera_id']}' experimenta un porcentaje de pérdida de paquetes significativamente mayor.")
    else:
        print("  No hay datos de resumen de pérdidas disponibles.")

    print("\nConclusión General:")
    print("  - Si una cámara muestra consistentemente latencias medias más altas en múltiples etapas, indica que los procesos para esa cámara son más lentos (RETRASADO).")
    print("  - Si una cámara muestra un porcentaje de pérdida de paquetes significativamente mayor, indica que ese camino está perdiendo/saltando más paquetes antes de ser contabilizados por seg_sub (SALTA PAQUETES).")
    print("  Ambos problemas pueden ocurrir simultáneamente.")


def plot_latencies_vs_iterations(df, output_dir="plots"):
    """Genera gráficas de latencias vs. packet_seq_num."""
    if df is None or df.empty:
        print("No hay datos para generar gráficas.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"\nGenerando gráficas en el directorio: {output_dir}")

    camera_ids = df['camera_id'].unique()
    
    latency_cols_to_plot = {
        "Latencia Total E2E (T0->T4)": "lat_total_e2e_ms",
        "Latencia T0->T1 (ImgPub a SegNode CB)": "lat_imgpub_to_cb_ms",
        "Latencia T1->T2 (SegNode CB a BatchStart)": "lat_cb_to_batch_start_ms",
        "Latencia T2->T3 (SegNode BatchStart a ResPub)": "lat_batch_start_to_res_pub_ms",
        "Latencia T3->T4 (SegNode ResPub a SegSub Recv)": "lat_res_pub_to_res_recv_ms"
    }

    for desc, col_name in latency_cols_to_plot.items():
        if col_name not in df.columns:
            print(f"Advertencia: Columna de latencia '{col_name}' no encontrada para plotear.")
            continue

        plt.figure(figsize=(15, 7))
        for camera_id in camera_ids:
            df_camera = df[df['camera_id'] == camera_id].copy()
            df_camera = df_camera.dropna(subset=[col_name, 'packet_seq_num'])
            if not df_camera.empty:
                plt.plot(df_camera['packet_seq_num'], df_camera[col_name], marker='.', linestyle='-', label=f'Cámara {camera_id}')
        
        plt.title(f'{desc} vs. Número de Secuencia del Paquete')
        plt.xlabel('Número de Secuencia del Paquete (packet_seq_num)')
        plt.ylabel('Latencia (ms)')
        plt.legend()
        plt.grid(True)
        plot_filename = os.path.join(output_dir, f"{col_name}_vs_seq_num.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"  Gráfica guardada: {plot_filename}")


def main():
    parser = argparse.ArgumentParser(description="Analiza el archivo CSV de log de latencias.")
    parser.add_argument("csv_filepath", type=str, help="Ruta al archivo latency_log.csv")
    parser.add_argument("--plot_dir", type=str, default="latency_plots", help="Directorio para guardar las gráficas.")
    args = parser.parse_args()

    df = load_data(args.csv_filepath)
    if df is None:
        return

    latency_intervals_map = calculate_and_print_latency_stats(df)
    loss_stats_summary_df = calculate_and_print_loss_stats(df)
    analyze_and_print_t0_offsets(df)
    
    if latency_intervals_map and loss_stats_summary_df is not None:
        evaluate_processing_performance(df, latency_intervals_map, loss_stats_summary_df)
    
    plot_latencies_vs_iterations(df, output_dir=args.plot_dir)

    print("\nAnálisis completado.")

if __name__ == "__main__":
    main()
