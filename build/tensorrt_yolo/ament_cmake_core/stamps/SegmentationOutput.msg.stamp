std_msgs/Header header
uint32 image_height        # Original height of the processed frame
uint32 image_width         # Original width of the processed frame

# Pixel-wise class ID map (flat array, row-major)
# Each value is the class ID for the corresponding pixel.
# 0 can be used for background or pixels with no assigned class.
int32[] class_id_map

# Pixel-wise instance ID map (flat array, row-major)
# Each value is a unique ID for the detected instance covering that pixel.
# 0 can be used for background or pixels with no assigned instance.
int32[] instance_id_map

# Confidence scores for each detected instance
float32[] instance_confidences

# Class IDs for each detected instance (corresponds to instance_confidences)
int32[] instance_class_ids

# Unique instance IDs assigned (corresponds to instance_confidences)
# These are the IDs used in instance_id_map
int32[] detected_instance_ids
