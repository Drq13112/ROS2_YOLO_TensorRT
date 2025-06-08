std_msgs/Header header         # Header con timestamp y frame_id
sensor_msgs/Image mask         # Máscara de segmentación. mask.data[i] = instance_id
                               # instance_id = 0 para fondo, 1 para la primera instancia, etc.
                               # mask.encoding será "mono8" o "mono16"
float32[] scores               # Array de scores, scores[j] es el score de la instancia j+1
int32[] classes                # Array de class_ids, classes[j] es la clase de la instancia j+1
                               # El tamaño de scores y classes debe ser igual al número de instancias detectadas.
# Opcional: podrías añadir también los bounding boxes aquí si los necesitas en el visualizador
# sensor_msgs/RegionOfInterest[] boxes # Por ejemplo