import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

# Carregando o dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
# Dividindo o dataset de treino em treino e validação de forma balanceada
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.25)
 
# Checando quantidade de imagens do dataset
print('Quantidade de imagens de treino:', x_train.shape[0])
print('Quantidade de imagens de validação:', x_val.shape[0])
print('Quantidade de imagens de test:', x_test.shape[0])

# Formatando o dataset para funcionar como entrada do Keras 
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Cada imagem precisa ter dimensão x, y e z

input_shape = (28, 28, 1)

# Convertento valores dos pixels para float (garantindo precisão em operações de divisão por exemplo)
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
 
# Normalizando os valores dos pixels (valores entre 0 e 1).
x_train /= 255
x_val /= 255
x_test /= 255

model = Sequential()

# Operação de convolução com filtro 3 x 3 seguida da função de ativação ReLU
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape, activation='relu'))

# Operação de Max Pooling 2 x 2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Operação de convolução com filtro 3 x 3 seguida da função de ativação ReLU
model.add(Conv2D(28, kernel_size=(3,3), activation='relu'))

# Mais uma camada de convolução
model.add(Conv2D(28, kernel_size=(3,3), activation='relu'))

# Operação de flatten (convertento o mapa de características em um vetor)
model.add(Flatten())

# Camada densa com 32 nerônios seguida da função de ativação ReLU
model.add(Dense(32, activation='relu'))

# Camada densa de saída com 10 (um para cada dígito) seguida de função SoftMax
model.add(Dense(10, activation='softmax'))

# Resumo do modelo
model.summary();

# Definindo otimizador, função de perda e métrica de eficiência.
adamOptimizer = Adam(learning_rate=0.001)
model.compile( optimizer=adamOptimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'] )

# Efetuando o treinamento de 4 épocas com o dataset de treino e validando no dataset de validação
history = model.fit(x=x_train, y=y_train, validation_data=(x_val,y_val), epochs=4, batch_size=16, shuffle=False)



# avaliando o modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = tf.argmax(y_pred_probs, axis=1)

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
print(f"Loss: {test_loss:.4f}")
print(f"Acurácia: {test_acc:.4f}")
print(f"Precision (macro): {macro_precision:.4f}")
print(f"Recall (macro): {macro_recall:.4f}")
print(f"Specificity (macro): {macro_specificity:.4f}")
print(f"F1-score (macro): {macro_f1:.4f}")

model.evaluate(x_test, y_test)

# Salvando o modelo em formato Keras .h5
path = "model.h5"
model.save(path)
print("Salvo (Keras .h5):", path)