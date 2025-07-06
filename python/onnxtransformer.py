import tensorflow as tf
import tf2onnx

# 1) Load your saved Sequential (weights + graph only, compile=False so it skips metrics)
base = tf.keras.models.load_model(
    "model_20250706-105237/v/best_20250706-175307.h5",
    compile=False
)

# 2) Build a proper Functional wrapper around it
#    We use the same input shape (drop the batch dimension None)
inp = tf.keras.Input(shape=base.input_shape[1:], name="input")
out = base(inp)
model = tf.keras.Model(inputs=inp, outputs=out, name="tetris_dqn")

# 3) Create the ONNX-compatible spec
spec = (tf.TensorSpec(model.input.shape, tf.float32, name="input"),)

# 4) Convert & save
output_path = "tetris_dqn.onnx"
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    output_path=output_path
)
print(f"ONNX model saved to {output_path}")
