import os
import cv2
import numpy as np
import glob
import time
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description="Recibe los parametros necesarios del numero de epochs y el dispositivo a utilizar")
parser.add_argument('--dispositivo',type=str, choices = ['cpu','gpu'], default = 'cpu')
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

	#Cargar el archivo de las etiquetas
	with open(path_labels, 'r') as f:
		labels = [line.strip() for line in f.readlines()]

	#Cargar las imagenes para test
	path_images = path + "/imagenes"
	images = glob.glob(path_images + '/*.jpg') + glob.glob(path_images + '/*.png') + glob.glob(path_images + '/*.bmp')
	images = sorted(images)

	loaded_model = tf.saved_model.load(path + "/saved_model_efficientdet_d0")
	infer = loaded_model.signatures['serving_default']
	times = []
	detections = []
	min_conf_threshold = 0.39

	def preprocess_image(image_path, target_size = (512,512)):
		image = cv2.imread(image_path)
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		imH, imW, _ = image.shape 
		image_resized = cv2.resize(image_rgb, target_size)
		#image_resized = image_resized / 255.0
		input_data = np.expand_dims(image_resized, axis=0)
		return input_data, imH, imW, image

	for image_path in images:
		input_image, imH, imW, image = preprocess_image(image_path)
		# Ejecutar la inferencia
		start_time = time.perf_counter()
		result = infer(tf.convert_to_tensor(input_image, dtype=tf.uint8))
		end_time = time.perf_counter()
		elapsed_time = end_time - start_time
		times.append(elapsed_time)
	    
		# Extraer los tensores del diccionario
		raw_detection_boxes =  result['raw_detection_boxes'].numpy()
		detection_multiclass_scores =  result['detection_multiclass_scores'].numpy()
		detection_classes =  result['detection_classes'][0].numpy()
		detection_boxes =  result['detection_boxes'][0].numpy()
		raw_detection_scores =  result['raw_detection_scores'].numpy()
		num_detections = int( result['num_detections'].numpy()[0])
		detection_anchor_indices =  result['detection_anchor_indices'].numpy()
		detection_scores =  result['detection_scores'][0].numpy()
	    
		for i in range(num_detections):
			if ((detection_scores[i] > min_conf_threshold) and (detection_scores[i] <= 1.0)):
				print(labels[int(detection_classes[i])])
				# Se obtienen las dimensiones de las cajas a dibujar
				ymin = int(max(1,(detection_boxes[i][0] * imH)))
				xmin = int(max(1,(detection_boxes[i][1] * imW)))
				ymax = int(min(imH,(detection_boxes[i][2] * imH)))
				xmax = int(min(imW,(detection_boxes[i][3] * imW)))

				cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

				# Dibuja las etiquetas en la imagen
				object_name = labels[int(detection_classes[i])]
				label = '%s: %d%%' % (object_name, int(detection_scores[i]*100))
				labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
				label_ymin = max(ymin, labelSize[1] + 10) 
				cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
				cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 	    
				detections.append([object_name, detection_scores[i], xmin, ymin, xmax, ymax])

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
