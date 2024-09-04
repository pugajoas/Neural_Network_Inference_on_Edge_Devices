import os
import glob
import cv2
import time
import numpy as np
import argparse
import tensorflow as tf

path = os.getcwd()

# Path para las etiquetas
path_labels = path + "/labels.txt"

#Cargar el archivo de las etiquetas
with open(path_labels, 'r') as f:
	labels = [line.strip() for line in f.readlines()]

loaded_model = tf.saved_model.load(path + "/saved_model_mobilenetv2_classification")
infer = loaded_model.signatures['serving_default']
times = []

def preprocess_image(image, target_size = (128,128)):
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	imH, imW, _ = image.shape 
	image_resized = cv2.resize(image_rgb, target_size)
	image_resized = image_resized / 255.0
	input_data = np.expand_dims(image_resized, axis=0)
	return input_data, image

# Abre la primera cámara (normalmente la cámara USB)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

frame_rate_calc = 1
# Obtener la frecuencia del reloj en ticks por segundo
freq = cv2.getTickFrequency()

# Inicializar el contador de frames
frame_count = 0

# Obtener el tiempo de inicio
t1 = cv2.getTickCount()

while True:
	
	# Captura frame por frame
	ret, frame = cap.read()
	frame_count += 1

    # Si el frame se capturó correctamente
	if not ret:
		print("No se pudo recibir el frame")
		break
	
	input_image, image = preprocess_image(frame)
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
	
	# Pone los fps en la esquina
	cv2.putText(image,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
	
	# Se muestra la imagen recibida con la deteccion realizada
	cv2.imshow('Object detector', image)
	
	# Calcular framerate
	t2 = cv2.getTickCount()
	time1 = (t2 - t1) / freq
	frame_rate_calc = frame_count / time1  # Calcula FPS
	
	total_time = sum(times)
	times_size = len(times)
	average_time = total_time / times_size
	average_time = average_time * 1000
	print(f"Tiempo de inferencia promedio: {average_time:.2f} ms") 
		
	# Sale del bucle si se presiona la tecla 'q'
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
            
# Limpia las ventanas abiertas por cv2
cv2.destroyAllWindows()


