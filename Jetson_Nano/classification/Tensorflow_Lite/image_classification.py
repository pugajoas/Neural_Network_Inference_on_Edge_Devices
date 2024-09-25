import os
import cv2
import numpy as np
import glob
import time
import argparse
from tensorflow.lite.python.interpreter import Interpreter

parser = argparse.ArgumentParser(description = "Recibe los parametros necesarios, show_images")
parser.add_argument('--no-show', action = 'store_false', dest = 'show', help = 'Mostrar la imagen detectada')
parser.add_argument('--save-results', action = 'store_true', dest = 'save', help = 'Guardar las imagenes de salida')
parser.set_defaults(show = True)
parser.set_defaults(save = False)
args = parser.parse_args()

path = os.getcwd()
path_model = path + "/mobilenetv2quantized.tflite"
#path_model = path + "/mobilenetv2.tflite"

if args.save:
	result_dir = 'results'
	result_path = os.path.join(path, result_dir)
	if not os.path.exists(result_path):
		os.makedirs(result_path)

# Path para las etiquetas
path_labels = path + "/labels.txt"

#Cargar las imagenes para test
path_images = path + "/imagenes"
images = glob.glob(path_images + '/*.jpg') + glob.glob(path_images + '/*.png') + glob.glob(path_images + '/*.bmp')
images = sorted(images)

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

def preprocess_image(image_path, width, height):
	image = cv2.imread(image_path)
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	imH, imW, _ = image.shape 
	image_resized = cv2.resize(image_rgb, (width, height))
	input_data = np.expand_dims(image_resized, axis=0)
	return input_data, image

for image_path in images:
	image = cv2.imread(image_path)
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	imH, imW, _ = image.shape 
	image_resized = cv2.resize(image_rgb, (width, height))
	input_data = np.expand_dims(image_resized, axis=0)
    
	if floating_model:
		input_data = (np.float32(input_data) - input_mean) / input_std
    
	# Establecer el tensor de entrada
	input_index = input_details[0]['index']
	interpreter.set_tensor(input_index, input_data)
    
	# Ejecutar la inferencia
	start_time = time.perf_counter()
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
        
	if args.show:
		# Se muestra la imagen recibida con la deteccion realizada
		cv2.imshow('Object detector', image)
            
		# Press any key to continue to next image, or press 'q' to quit
		if cv2.waitKey(0) == ord('q'):
			break
        
	if args.save:
		image_name = os.path.basename(image_path)
		image_savepath = os.path.join(path,result_dir,image_name)
		cv2.imwrite(image_savepath, image)

total_time = sum(times)
times_size = len(times)
average_time = total_time / times_size
average_time = average_time * 1000
print(f"Tiempo de inferencia promedio: {average_time:.2f} ms")
