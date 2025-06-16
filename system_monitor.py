import psutil
import pynvml # Para GPUs NVIDIA
import time
import csv
import argparse
import datetime
import os

def get_gpu_info(handle):
    """Obtiene información de una GPU NVIDIA específica."""
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0 # Convertir mW a W
        
        return {
            "gpu_util_percent": util.gpu,
            "gpu_mem_util_percent": util.memory,
            "gpu_mem_used_mb": memory.used / (1024**2),
            "gpu_mem_total_mb": memory.total / (1024**2),
            "gpu_mem_free_mb": memory.free / (1024**2),
            "gpu_temp_c": temp,
            "gpu_power_w": power_usage
        }
    except pynvml.NVMLError as e:
        print(f"Error al obtener información de la GPU: {e}")
        return {
            "gpu_util_percent": None, "gpu_mem_util_percent": None,
            "gpu_mem_used_mb": None, "gpu_mem_total_mb": None, "gpu_mem_free_mb": None,
            "gpu_temp_c": None, "gpu_power_w": None
        }

def find_processes_by_name(names):
    """Encuentra PIDs de procesos que coincidan con una lista de nombres."""
    pids = {}
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for name_to_find in names:
                if name_to_find.lower() in proc.info['name'].lower():
                    if name_to_find not in pids:
                        pids[name_to_find] = []
                    pids[name_to_find].append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return pids

def get_process_cpu_usage(pids_dict):
    """Obtiene el uso de CPU para los PIDs dados."""
    process_cpu_usages = {}
    for name, pid_list in pids_dict.items():
        total_cpu_for_name = 0
        valid_pids_for_name = 0
        for pid in pid_list:
            try:
                p = psutil.Process(pid)
                total_cpu_for_name += p.cpu_percent(interval=None) 
                valid_pids_for_name +=1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass 
        if valid_pids_for_name > 0:
            process_cpu_usages[f"proc_{name}_cpu_percent_sum"] = total_cpu_for_name
        else:
            process_cpu_usages[f"proc_{name}_cpu_percent_sum"] = None
    return process_cpu_usages


def main():
    parser = argparse.ArgumentParser(description="Monitoriza el uso de CPU, GPU, RAM y Red, y lo guarda en un CSV.")
    parser.add_argument("--interval", type=float, default=1.0, help="Intervalo de muestreo en segundos (ej: 0.02 para 20ms).")
    parser.add_argument("--duration", type=float, default=None, help="Duración total de la monitorización en segundos (opcional).")
    parser.add_argument("--output_file", type=str, default="system_metrics_log.csv", help="Archivo CSV de salida.")
    parser.add_argument("--process_names", nargs='*', default=[], help="Nombres de procesos para monitorizar su CPU (ej: seg_sub tensorrt_yolo).")
    args = parser.parse_args()

    gpu_handles = []
    num_gpus = 0
    try:
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
        for i in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_handles.append(handle)
        print(f"NVML inicializado. Se encontraron {num_gpus} GPUs NVIDIA.")
    except pynvml.NVMLError as e:
        print(f"No se pudo inicializar NVML: {e}. La monitorización de GPU estará deshabilitada.")

    fieldnames = ["timestamp_epoch", "datetime_utc"]
    fieldnames.extend([
        "cpu_total_percent", "system_ram_used_mb", "system_ram_used_percent",
        "net_sent_mbps", "net_recv_mbps" # Nuevas columnas para ancho de banda
    ])
    
    for proc_name_arg in args.process_names:
        clean_proc_name = "".join(c if c.isalnum() else "_" for c in proc_name_arg)
        fieldnames.append(f"proc_{clean_proc_name}_cpu_percent_sum")

    for i in range(num_gpus):
        fieldnames.extend([
            f"gpu{i}_util_percent", f"gpu{i}_mem_util_percent",
            f"gpu{i}_mem_used_mb", f"gpu{i}_mem_total_mb", f"gpu{i}_mem_free_mb",
            f"gpu{i}_temp_c", f"gpu{i}_power_w"
        ])

    print(f"Guardando métricas en: {args.output_file}")
    print(f"Intervalo de muestreo: {args.interval}s")
    if args.duration:
        print(f"Duración de la monitorización: {args.duration}s")
    if args.process_names:
        print(f"Monitorizando CPU para procesos que contengan: {', '.join(args.process_names)}")

    if args.process_names:
        initial_pids_dict = find_processes_by_name(args.process_names)
        for name, pid_list in initial_pids_dict.items():
            for pid in pid_list:
                try:
                    psutil.Process(pid).cpu_percent(interval=0.01) 
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        time.sleep(0.1) 

    start_time = time.time()
    # Inicializar contadores de red para la primera medición de ancho de banda
    last_net_io = psutil.net_io_counters()
    last_net_time = start_time

    try:
        with open(args.output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while True:
                loop_start_time = time.time() # Tiempo al inicio del bucle actual

                if args.duration and (loop_start_time - start_time) > args.duration:
                    print("Duración de monitorización alcanzada. Terminando.")
                    break

                epoch_time = loop_start_time
                utc_datetime = datetime.datetime.utcnow().isoformat()
                
                row_data = {
                    "timestamp_epoch": epoch_time,
                    "datetime_utc": utc_datetime
                }

                # Uso total de CPU
                row_data["cpu_total_percent"] = psutil.cpu_percent(interval=None)

                # Memoria RAM del sistema (CPU)
                sys_vmem = psutil.virtual_memory()
                row_data["system_ram_used_mb"] = sys_vmem.used / (1024**2)
                row_data["system_ram_used_percent"] = sys_vmem.percent

                # Ancho de banda de Red
                current_net_io = psutil.net_io_counters()
                current_net_time = epoch_time # Usar el timestamp actual del bucle

                # Calcular el tiempo transcurrido desde la última medición de red
                time_diff_net = current_net_time - last_net_time
                
                if time_diff_net > 0: # Evitar división por cero si el intervalo es muy rápido
                    bytes_sent_diff = current_net_io.bytes_sent - last_net_io.bytes_sent
                    bytes_recv_diff = current_net_io.bytes_recv - last_net_io.bytes_recv

                    # Convertir bytes/segundo a Megabits/segundo (Mbps)
                    # 1 byte = 8 bits; 1 Megabit = 10^6 bits
                    row_data["net_sent_mbps"] = (bytes_sent_diff * 8) / (time_diff_net * 1000000)
                    row_data["net_recv_mbps"] = (bytes_recv_diff * 8) / (time_diff_net * 1000000)
                else:
                    row_data["net_sent_mbps"] = 0.0
                    row_data["net_recv_mbps"] = 0.0
                
                last_net_io = current_net_io
                last_net_time = current_net_time


                if args.process_names:
                    pids_dict = find_processes_by_name(args.process_names)
                    process_cpu_data = get_process_cpu_usage(pids_dict)
                    for proc_name_arg in args.process_names:
                        clean_proc_name = "".join(c if c.isalnum() else "_" for c in proc_name_arg)
                        key_in_data = f"proc_{proc_name_arg}_cpu_percent_sum"
                        csv_col_name = f"proc_{clean_proc_name}_cpu_percent_sum"
                        row_data[csv_col_name] = process_cpu_data.get(key_in_data, None)

                for i, handle in enumerate(gpu_handles):
                    gpu_info = get_gpu_info(handle)
                    row_data[f"gpu{i}_util_percent"] = gpu_info["gpu_util_percent"]
                    row_data[f"gpu{i}_mem_util_percent"] = gpu_info["gpu_mem_util_percent"]
                    row_data[f"gpu{i}_mem_used_mb"] = gpu_info["gpu_mem_used_mb"]
                    row_data[f"gpu{i}_mem_total_mb"] = gpu_info["gpu_mem_total_mb"]
                    row_data[f"gpu{i}_mem_free_mb"] = gpu_info["gpu_mem_free_mb"]
                    row_data[f"gpu{i}_temp_c"] = gpu_info["gpu_temp_c"]
                    row_data[f"gpu{i}_power_w"] = gpu_info["gpu_power_w"]
                
                writer.writerow(row_data)

                elapsed_in_loop = time.time() - loop_start_time
                sleep_time = args.interval - elapsed_in_loop
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # else:
                #     if args.interval > 0: 
                #         print(f"Advertencia: El bucle de monitorización tardó más ({elapsed_in_loop:.4f}s) que el intervalo especificado ({args.interval}s).")

    except KeyboardInterrupt:
        print("\nMonitorización interrumpida por el usuario.")
    except Exception as e:
        print(f"Ocurrió un error durante la monitorización: {e}")
    finally:
        if num_gpus > 0:
            try:
                pynvml.nvmlShutdown()
                print("NVML apagado.")
            except pynvml.NVMLError as e:
                print(f"Error al apagar NVML: {e}")
        print(f"Datos de monitorización guardados en {args.output_file}")

if __name__ == "__main__":
    main()