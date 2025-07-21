std_msgs/Header header
sensor_msgs/Image mask
float32[] scores
int32[] classes

builtin_interfaces/Time image_source_monotonic_capture_time # T0 (SegNode Callback Entry)
builtin_interfaces/Time processing_node_monotonic_entry_time # T1 (SegNode Batch Processing Start)
builtin_interfaces/Time processing_node_inference_start_time # T2a (SegNode Inference Start)
builtin_interfaces/Time processing_node_inference_end_time   # T2b (SegNode Inference End)
builtin_interfaces/Time processing_node_monotonic_publish_time # T3 (SegNode Result Publish)
uint64 packet_sequence_number # Corresponds to source_image_seq from TimedImage
