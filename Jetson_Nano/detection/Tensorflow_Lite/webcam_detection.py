import os
import cv2
import numpy as np
import glob
import time
import argparse
from tensorflow.lite.python.interpreter import Interpreter

parser = argparse.ArgumentParser(description = "Recibe los parametros necesarios, show_images")
parser.add_argument('--no-show', action = 'store_false', dest = 'show', help = 'Mostrar la imagen detectada')
parser.set_defaults(show = True)
args = parser.parse_args()

path = os.getcwd()
path_model = path + "/efficientdet_lite0.tflite"

# Path para las etiquetas
path_labels = path + "/labels.txt"

#Cargar el archivo de las etiquetas
with open(path_labels, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

interpreter = Interpreter(model_path=path_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

outname = output_details[0]['name']

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']
times = []

def preprocess_image(image, width, height):
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	imH, imW, _ = image.shape
	#image_resized = image_resized / 255.0
	image_resized = cv2.resize(image_rgb, (width, height))
	input_data = np.expand_dims(image_resized, axis=0)
	return input_data, imH, imW, image

detections = []
min_conf_threshold = 0.5

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

	input_image, imH, imW, image = preprocess_image(frame, width, height)
    
	if floating_model:
		input_data = (np.float32(input_data) - input_mean) / input_std
		
	# Establecer el tensor de entrada
	input_index = input_details[0]['index']
	interpreter.set_tensor(input_index, input_image)
    
	# Ejecutar la inferencia
	start_time = time.perf_counter()
	interpreter.invoke()
	end_time = time.perf_counter()
	elapsed_time = end_time - start_time
	times.append(elapsed_time)

	# Divide los resultados obtenidos por el modelo
	detection_classes =  interpreter.get_tensor(output_details[1]['index'])[0]
	detection_boxes =  interpreter.get_tensor(output_details[0]['index'])[0]
	num_detections = int(interpreter.get_tensor(output_details[3]['index'])[0])
	detection_scores =  interpreter.get_tensor(output_details[2]['index'])[0]

	# Loop entre todos los resultados obtenidos donde toma en cuenta los valores arriba del threshold
	for i in range(num_detections):
		if ((detection_scores[i] > min_conf_threshold) and (detection_scores[i] <= 1.0)):

			# Se obtienen las dimensiones de las cajas a dibujar
				ymin = int(detection_boxes[i][0] * imH)
				xmin = int(detection_boxes[i][1] * imW)
				ymax = int(detection_boxes[i][2] * imH)
				xmax = int(detection_boxes[i][3] * imW)
            
				cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

				# Dibuja las etiquetas en la imagen
				object_name = labels[int(detection_classes[i])]
				label = '%s: %d%%' % (object_name, int(detection_scores[i]*100))
				labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
				label_ymin = max(ymin, labelSize[1] + 10) 
				cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
				cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 	    
				detections.append([object_name, detection_scores[i], xmin, ymin, xmax, ymax])
	
	# Pone los fps en la esquina
	cv2.putText(image,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA) 
        
	# Se muestra la imagen recibida con la deteccion realizada
	cv2.imshow('Object detector', image)

	# Calcular framerate
	t2 = cv2.getTickCount()
	time1 = (t2 - t1) / freq
	frame_rate_calc = frame_count / time1  # Calcula FPS

	#total_time = sum(times)
	#times_size = len(times)
	#average_time = total_time / times_size
	#average_time = average_time * 1000
	#print(f"Tiempo de inferencia promedio: {average_time:.2f} ms")

	# Sale del bucle si se presiona la tecla 'q'
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


# Limpia las ventanas abiertas por cv2
cv2.destroyAllWindows()


