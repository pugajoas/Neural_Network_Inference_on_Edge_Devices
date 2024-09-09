import os
import cv2
import numpy as np
import glob
import time
import argparse
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

parser = argparse.ArgumentParser(description="Recibe los parametros necesarios del numero de epochs y el dispositivo a utilizar")
parser.add_argument('--dispositivo',type=str, choices = ['cpu','gpu'], default = 'cpu')
parser.set_defaults(show = True)
args = parser.parse_args()
dispositivo = args.dispositivo

print(tf.config.list_physical_devices('GPU'))

path = os.getcwd()
path_model = path + "/mobilenetv2quantized.tflite"
#path_model = path + "/mobilenetv2.tflite"

# Path para las etiquetas
path_labels = path + "/labels.txt"

#Cargar el archivo de las etiquetas
with open(path_labels, 'r') as f:
	labels = [line.strip() for line in f.readlines()]

interpreter = Interpreter(model_path=path_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

times = []

def preprocess_image(image, width, height):
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	imH, imW, _ = image.shape 
	image_resized = cv2.resize(image_rgb, (width, height))
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

# Definir la resolución deseada
camara_width = 640
camara_height = 480

# Establecer la resolución de la cámara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camara_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camara_height)

while True:

	# Captura frame por frame
	ret, frame = cap.read()
	frame_count += 1

    	# Si el frame se capturó correctamente
	if not ret:
		print("No se pudo recibir el frame")
		break	

	input_data, image = preprocess_image(frame, width, height)
    
	if floating_model:
		input_data = (np.float32(input_data) - input_mean) / input_std
    
	# Establecer el tensor de entrada
	input_index = input_details[0]['index']
	interpreter.set_tensor(input_index, input_data)
    
	# Ejecutar la inferencia
	start_time = time.perf_counter()
	with tf.device(dispositivo):
		interpreter.invoke()
	end_time = time.perf_counter()
	elapsed_time = end_time - start_time
	times.append(elapsed_time)

	# Obtener los resultados del tensor de salida
	output_data = interpreter.get_tensor(output_details[0]['index'])
	# Si el modelo realiza clasificación y quieres la clase con la mayor puntuación
	predicted_class_index = np.argmax(output_data)
	if floating_model:
		predicted_class_probability = output_data[0][predicted_class_index] * 100
	else:
		predicted_class_probability = (output_data[0][predicted_class_index] / 255) * 100

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

        
	#print(f"Prediccion: {labels[predicted_class]} con {percentage:.2f}%")

