import os
import tensorflow as tf 
import sys
if sys.version_info.major >= 3:
    import pathlib
else:
    import pathlib2 as pathlib
models_path = os.path.join(os.getcwd(), "models")
sys.path.append(models_path)
saved_models_root = '/Users/shrey/Desktop/model'
saved_model_dir = str(sorted(pathlib.Path(saved_models_root).glob("*"))[-1])

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
tflite_model_file = "/Users/shrey/Desktop/litemodel/mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)