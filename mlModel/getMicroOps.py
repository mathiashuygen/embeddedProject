import tensorflow as tf

interpreter = tf.lite.Interpreter("model.tflite")
interpreter.allocate_tensors()

details = interpreter._get_ops_details()
ops = set(op["op_name"] for op in details)
for op in sorted(ops):
    print(op)
