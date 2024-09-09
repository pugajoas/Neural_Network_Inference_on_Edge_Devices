import os
import glob
import cv2
import time
import numpy as np
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description="Recibe los parametros necesarios del numero de epochs y el dispositivo a utilizar")
parser.add_argument('--dispositivo',type=str, choices = ['cpu','gpu'], default = 'cpu')
parser.add_argument('--no-show', action = 'store_false', dest = 'show', help = 'Mostrar la imagen detectada')
parser.set_defaults(show = True)
args = parser.parse_args()
dispositivo = args.dispositivo

if dispositivo == 'gpu' and tf.config.list_physical_devices('GPU'):
	dispositivo = '/gpu:0'
else:
	dispositivo = '/cpu:0'

with tf.device(dispositivo):
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
		return input_data, image

	for image_path in images:
		input_image, image = preprocess_image(image_path)
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
		#print(f"Prediccion: {labels[predicted_class_index]} con {predicted_class_probability:.2f}%")
		label = f"{labels[predicted_class_index]} {predicted_class_probability:.2f}%"
		cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) 
		
		if args.show:
			# Se muestra la imagen recibida con la deteccion realizada
			cv2.imshow('Object detector', image)
	    
			# Press any key to continue to next image, or press 'q' to quit
			if cv2.waitKey(0) == ord('q'):
				break

	total_time = sum(times)
	times_size = len(times)
	average_time = total_time / times_size
	average_time = average_time * 1000
	print(f"Tiempo de inferencia promedio: {average_time:.2f} ms")
