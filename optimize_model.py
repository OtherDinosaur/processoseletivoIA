import tensorflow as tf
import os
import os, time, importlib, numpy as np
from tensorflow import keras

# Preparando o dataset para avaliação do modelo otimizado
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = (x_train.astype("float32") / 255.0)[..., None]
x_test  = (x_test.astype("float32") / 255.0)[..., None]

print("x_train:", x_train.shape, x_train.dtype)
print("x_test :", x_test.shape,  x_test.dtype)
print("y_train:", y_train.shape, " | y_test:", y_test.shape)


# Carregando o modelo salvo em formato Keras .h5
model = tf.keras.models.load_model("model.h5")
print("Modelo carregado com sucesso!")

keras_path = "model.h5"


# (B) TFLite dynamic range — quantiza pesos; I/O permanece float32
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantização pós-treinamento (intervalo dinâmico)
tflite_dynamic = converter.convert()
tfl_dyn_path = "teste_dynamic.tflite"
with open(tfl_dyn_path, "wb") as f:
    f.write(tflite_dynamic)
print("Gerado:", tfl_dyn_path)


try:
    if importlib.util.find_spec("ai_edge_litert") is None:
        import sys, subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ai-edge-litert"])
    from ai_edge_litert.interpreter import Interpreter   # novo interpretador LiteRT
    _use_litert = True
except Exception as e:
    # fallback para ambientes sem o pacote ou TF < 2.20
    from tensorflow.lite import Interpreter
    _use_litert = False
    print("Usando tf.lite.Interpreter (fallback). Mensagem:", e)

print("Interpreter em uso:", "LiteRT (ai-edge-litert)" if _use_litert else "tf.lite (TensorFlow)")



def run_single_inference(tflite_path, img):
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], img.astype("float32"))
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details["index"])[0]
    pred = int(np.argmax(probs))
    return probs, pred

# Pegar 1 amostra do conjunto de teste
i = 0
img = x_test[i:i+1]  # (1,28,28,1)

probs_dyn, pred_dyn = run_single_inference(tfl_dyn_path, img)
probs_keras = model.predict(img, verbose=0)[0]
pred_keras = int(np.argmax(probs_keras))

print("🔎 Resultado da previsão para a imagem", i)
print("="*50)
print(f"📌 Resposta correta (rótulo): {y_test[i]}")
print(f"🤖 Keras previu       → {pred_keras}")
print(f"⚡ LiteRT (dynamic)   → {pred_dyn}")
print("-"*50)


def size_mb(p):
    return os.path.getsize(p) / (1024 * 1024)

print("\nTamanhos aproximados:")
print(f" - {keras_path}:    {size_mb(keras_path):.2f} MB (HDF5 legado)")
print(f" - {tfl_dyn_path}:     {size_mb(tfl_dyn_path):.2f} MB (TFLite dynamic)")




def create_interpreter(tflite_path):
    interp = Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    input_details = interp.get_input_details()[0]
    output_details = interp.get_output_details()[0]
    return interp, input_details, output_details

def warmup(interpreter, input_details, output_details, img, runs=5):
    for _ in range(runs):
        interpreter.set_tensor(input_details["index"], img)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])

def benchmark(interpreter, input_details, output_details, img, runs=100):
    start = time.perf_counter()
    for _ in range(runs):
        interpreter.set_tensor(input_details["index"], img)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])
    end = time.perf_counter()
    total = (end - start) * 1000.0  # ms
    return total / runs

# Preparar input único
bench_img = x_test[:1].astype("float32")

# dynamic
interp_dyn, in_dyn, out_dyn = create_interpreter(tfl_dyn_path)
warmup(interp_dyn, in_dyn, out_dyn, bench_img, runs=10)
lat_dyn_ms = benchmark(interp_dyn, in_dyn, out_dyn, bench_img, runs=100)

print(f"Latência média (100 execuções):")
print(f" - dynamic : {lat_dyn_ms:.3f} ms")