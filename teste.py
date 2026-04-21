# Código correto para carregar o modelo TFLite (teste.py)
import tensorflow as tf
import numpy as np
from tensorflow import keras

# ✅ FORMA CORRETA - Carregando modelo TFLite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Obtendo informações de entrada e saída
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])

# Carregando dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = (x_test.astype('float32') / 255.0)[..., None]

print(f"\nTestando modelo em {len(x_test)} imagens...")

# Fazendo inferências em todo o dataset de teste
y_pred = []
for i, test_image in enumerate(x_test):
    test_image_batch = np.expand_dims(test_image, axis=0).astype('float32')
    interpreter.set_tensor(input_details[0]['index'], test_image_batch)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    y_pred.append(np.argmax(output_data[0]))
    
    if (i + 1) % 2000 == 0:
        print(f"Processadas {i + 1} imagens...")

y_pred = np.array(y_pred)
print(f"✅ Inferência completa!")

# Calculando acurácia geral
accuracy = np.mean(y_pred == y_test)
print(f"\n=== ACURÁCIA GERAL ===")
print(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Calculando a matriz de confusão
cm = tf.math.confusion_matrix(y_test, y_pred)
print("\n=== MATRIZ DE CONFUSÃO ===")
print(cm.numpy())

cm = tf.cast(cm, tf.float32)

tp = tf.linalg.diag_part(cm)
fp = tf.reduce_sum(cm, axis=0) - tp
fn = tf.reduce_sum(cm, axis=1) - tp
tn = tf.reduce_sum(cm) - (tp + fp + fn)

# Criando métricas por classe
precision = tp / (tp + fp + 1e-7)
recall    = tp / (tp + fn + 1e-7)
specificity = tn / (tn + fp + 1e-7)
f1_score  = 2 * precision * recall / (precision + recall + 1e-7)
accuracy_per_class = (tp + tn) / (tp + tn + fp + fn + 1e-7)

print("\n=== MÉTRICAS POR CLASSE ===")
for i in range(10):
    print(
        f"Class {i}: "
        f"Precision={precision[i]:.4f} | "
        f"Recall={recall[i]:.4f} | "
        f"Specificity={specificity[i]:.4f} | "
        f"F1={f1_score[i]:.4f} | "
        f"Acc={accuracy_per_class[i]:.4f}"
    )

# Calculando métricas gerais (média das métricas por classe)
macro_precision = tf.reduce_mean(precision)
macro_recall = tf.reduce_mean(recall)
macro_specificity = tf.reduce_mean(specificity)
macro_f1 = tf.reduce_mean(f1_score)

print("\n=== MÉTRICAS GERAIS ===")
print(f"Precision (macro): {macro_precision:.4f}")
print(f"Recall (macro): {macro_recall:.4f}")
print(f"Specificity (macro): {macro_specificity:.4f}")
print(f"F1-score (macro): {macro_f1:.4f}")