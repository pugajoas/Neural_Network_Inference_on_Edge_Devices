import os
import cv2
import numpy as np
import glob
import time
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description = "Recibe los parametros necesarios, show_images")
parser.add_argument('--no-show', action = 'store_false', dest = 'show', help = 'Mostrar la imagen detectada')
parser.set_defaults(show = True)
args = parser.parse_args()

path = os.getcwd()

cyan = (255, 255, 0)
magenta = (255, 0, 255)

EDGE_COLORS = {
	(0, 1): magenta,
	(0, 2): cyan,
	(1, 3): magenta,
	(2, 4): cyan,
	(0, 5): magenta,
	(0, 6): cyan,
	(5, 7): magenta,
	(7, 9): cyan,
	(6, 8): magenta,
	(8, 10): cyan,
	(5, 6): magenta,
	(5, 11): cyan,
	(6, 12): magenta,
	(11, 12): cyan,
	(11, 13): magenta,
	(13, 15): cyan,
	(12, 14): magenta,
	(14, 16): cyan
}

width = 192
height = 192

loaded_model = tf.saved_model.load(path + "/saved_model_singlepose_lightning")
infer = loaded_model.signatures['serving_default']
times = []

def preprocess_image(image, target_size = (192,192)):
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	imH, imW, _ = image.shape 
	image_resized = cv2.resize(image_rgb, target_size)
	#image_resized = image_resized / 255.0
	input_data = np.expand_dims(image_resized, axis=0)
	return input_data, imH, imW, image, image_resized

def loop(frame, keypoints, threshold=0.11):
	"""
	Main loop : Draws the keypoints and edges for each instance
	"""
	
	# Loop through the results
	for instance in keypoints: 
		# Draw the keypoints and get the denormalized coordinates
		denormalized_coordinates = draw_keypoints(frame, instance, threshold)
		# Draw the edges
		draw_edges(denormalized_coordinates, frame, EDGE_COLORS, threshold)
	
def draw_keypoints(frame, keypoints, threshold=0.11):
	"""Draws the keypoints on a image frame"""    
	# Denormalize the coordinates : multiply the normalized coordinates by the input_size(width,height)
	denormalized_coordinates = np.squeeze(np.multiply(keypoints, [width,height,1]))
	#Iterate through the points
	for keypoint in denormalized_coordinates:
		# Unpack the keypoint values : y, x, confidence score
		keypoint_y, keypoint_x, keypoint_confidence = keypoint
		if keypoint_confidence > threshold:
			""""
			Draw the circle
			Note : A thickness of -1 px will fill the circle shape by the specified color.
			"""
			cv2.circle(
				img=frame, 
			center=(int(keypoint_x), int(keypoint_y)), 
			radius=4, 
			color=(255,0,0),
			thickness=-1
			)
	return denormalized_coordinates

def draw_edges(denormalized_coordinates, frame, edges_colors, threshold=0.11):
	"""
	Draws the edges on a image frame
	"""
	
	# Iterate through the edges 
	for edge, color in edges_colors.items():
		# Get the dict value associated to the actual edge
		p1, p2 = edge
		# Get the points
		y1, x1, confidence_1 = denormalized_coordinates[p1]
		y2, x2, confidence_2 = denormalized_coordinates[p2]
		# Draw the line from point 1 to point 2, the confidence > threshold
		if (confidence_1 > threshold) & (confidence_2 > threshold):      
			cv2.line(img=frame, pt1=(int(x1), int(y1)),pt2=(int(x2), int(y2)), color=color, thickness=2, 
			lineType=cv2.LINE_AA # Gives anti-aliased (smoothed) line which looks great for curves
			)
		
def resize_back(image_resized, original_shape):
	"""
	Redimensiona una imagen de vuelta a sus dimensiones originales.   
	Args:
		image_resized (numpy.ndarray): La imagen redimensionada.
		original_shape (tuple): Las dimensiones originales de la imagen en (altura, ancho).
	
	Returns:
		numpy.ndarray: La imagen redimensionada de vuelta a sus dimensiones originales.
	"""
	original_height, original_width = original_shape
	# Redimensionar la imagen redimensionada de vuelta a las dimensiones originales
	image_back_to_original = cv2.resize(image_resized, (original_width, original_height))
	return image_back_to_original	
	

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
	
	input_image, imH, imW, image, image_resized = preprocess_image(frame)
	
	# Ejecutar la inferencia
	start_time = time.perf_counter()
	result = infer(tf.convert_to_tensor(input_image, dtype=tf.int32))
	end_time = time.perf_counter()
	elapsed_time = end_time - start_time
	times.append(elapsed_time)
	loop(image_resized, result['output_0'],threshold=0.11)
	
	# Reescalar coordenadas (si es necesario)
	image_back_to_original = resize_back(image_resized, (imH, imW))
	image_back_to_original_bgr = cv2.cvtColor(image_back_to_original, cv2.COLOR_RGB2BGR)
	
	# Pone los fps en la esquina
	cv2.putText(image_back_to_original_bgr,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

	cv2.imshow('Video en Vivo',image_back_to_original_bgr)
	
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

