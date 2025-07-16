std_msgs/Header header

# El mapa de segmentación combinado.
# Encoding esperado: "8UC2"
# Canal 0: ID de la clase (0-19)
# Canal 1: Confianza del píxel (0-255)
sensor_msgs/Image segmentation_map

# Número de secuencia para seguimiento de lotes y pérdidas
uint64 packet_sequence_number

# Timestamps para análisis de latencia (usando el reloj MONOTONIC del nodo de procesamiento)
builtin_interfaces/Time image_source_monotonic_capture_time   # T1: Entrada al callback de la imagen
builtin_interfaces/Time processing_node_monotonic_entry_time  # T2: Inicio del procesamiento del lote
builtin_interfaces/Time processing_node_inference_start_time  # T2a: Inicio de la inferencia
builtin_interfaces/Time processing_node_inference_end_time    # T2b: Fin de la inferencia
builtin_interfaces/Time processing_node_monotonic_publish_time  # T3: Publicación del resultado