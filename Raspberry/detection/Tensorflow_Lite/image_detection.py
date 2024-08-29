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

#Cargar las imagenes para test
path_images = path + "/imagenes"
images = glob.glob(path_images + '/*.jpg') + glob.glob(path_images + '/*.png') + glob.glob(path_images + '/*.bmp')
images = sorted(images)

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

def preprocess_image(image_path, width, height):
	image = cv2.imread(image_path)
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	imH, imW, _ = image.shape
	#image_resized = image_resized / 255.0
	image_resized = cv2.resize(image_rgb, (width, height))
	input_data = np.expand_dims(image_resized, axis=0)
	return input_data, imH, imW, image

for image_path in images:
	input_image, imH, imW, image = preprocess_image(image_path, width, height)
    
	if floating_model:
		input_image = (np.float32(input_image) - input_mean) / input_std
		
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
	boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
	classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
	scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

	detections = []
	min_conf_threshold = 0.5

	# Loop entre todos los resultados obtenidos donde toma en cuenta los valores arriba del threshold
	for i in range(len(scores)):
		if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

			# Se obtienen las dimensiones de las cajas a dibujar
			ymin = int(max(1,(boxes[i][0] * imH)))
			xmin = int(max(1,(boxes[i][1] * imW)))
			ymax = int(min(imH,(boxes[i][2] * imH)))
			xmax = int(min(imW,(boxes[i][3] * imW)))
            
			cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

			# Dibuja las etiquetas en la imagen
			object_name = labels[int(classes[i])]
			label = '%s: %d%%' % (object_name, int(scores[i]*100))
			labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
			label_ymin = max(ymin, labelSize[1] + 10) 
			cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
			cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 

			detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
			
	if args.show:
		# Se muestra la imagen recibida con la deteccion realizada
		cv2.imshow('Object detector', image)
        
		# Press any key to continue to next image, or press 'q' to quit
		if cv2.waitKey(0) == ord('q'):
			break

# Limpia las ventanas abiertas por cv2
cv2.destroyAllWindows()

del times[0]
total_time = sum(times)
times_size = len(times)
average_time = total_time / times_size
average_time = average_time * 1000
print(f"Tiempo de inferencia promedio: {average_time:.2f} ms")

