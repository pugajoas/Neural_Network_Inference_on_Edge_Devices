import os
import glob
import cv2
import time
import numpy as np
import tensorflow as tf


path = os.getcwd()

# Path para las etiquetas
path_labels = path + "/labels.txt"

#Cargar las imagenes para test
path_images = path + "/imagenes"
images = glob.glob(path_images + '/*.jpg') + glob.glob(path_images + '/*.png') + glob.glob(path_images + '/*.bmp')
images = sorted(images)

#Cargar el archivo de las etiquetas
with open(path_labels, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

loaded_model = tf.saved_model.load(path + "/saved_model_mobilenetv2_classification")
infer = loaded_model.signatures['serving_default']
times = []

def preprocess_image(image_path, target_size = (128,128)):
	image = cv2.imread(image_path)
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	imH, imW, _ = image.shape 
	image_resized = cv2.resize(image_rgb, target_size)
	image_resized = image_resized / 255.0
	input_data = np.expand_dims(image_resized, axis=0)
	return input_data

for image_path in images:
	input_image = preprocess_image(image_path)
	# Ejecutar la inferencia
	start_time = time.perf_counter()
	result = infer(tf.convert_to_tensor(input_image, dtype=tf.float32))
	end_time = time.perf_counter()
	elapsed_time = end_time - start_time
	times.append(elapsed_time)

	probabilities = tf.nn.softmax(result['logits'])

	# Obtener las probabilidades de la imagen
	probabilities = probabilities.numpy()[0]

	# Encontrar el índice de la clase con la probabilidad más alta
	predicted_class_index = np.argmax(probabilities)
	predicted_class_probability = probabilities[predicted_class_index] * 100
	print(f"Prediccion: {labels[predicted_class_index]} con {predicted_class_probability:.2f}%")

total_time = sum(times)
times_size = len(times)
average_time = total_time / times_size
average_time = average_time * 1000
print(f"Tiempo de inferencia promedio: {average_time:.2f} ms")
