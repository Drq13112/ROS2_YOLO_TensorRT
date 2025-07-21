import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("yolo11m-seg.onnx")
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print("Input name:", input_name)
print("Input shape:", input_shape)

dummy_input = np.random.rand(1, 3, 1216, 5760).astype(np.float32)
outputs = session.run(None, {input_name: dummy_input})
for i, out in enumerate(outputs):
    print(f"Output {i}: shape {out.shape}")
