import tensorflow as tf
import os

#insira seu código aqui

# Carregando o modelo salvo em formato Keras .h5
model = tf.keras.models.load_model("model.h5")
print("Modelo carregado com sucesso!")

