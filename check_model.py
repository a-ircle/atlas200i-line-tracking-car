import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1 


ort_session = ort.InferenceSession("yolov5s.onnx", sess_options)
print("Model loaded successfully")
