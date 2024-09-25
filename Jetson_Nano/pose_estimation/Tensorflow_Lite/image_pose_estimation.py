import os
import glob
import cv2
import numpy as np
import time
import argparse
from tensorflow.lite.python.interpreter import Interpreter
import tensorflow as tf 

parser = argparse.ArgumentParser(description="Recibe el parametro para no mostrar las imagenes procesadas")
parser.add_argument('--no-show', action = 'store_false', dest = 'show', help = 'Mostrar la imagen detectada')
parser.add_argument('--save-results', action = 'store_true', dest = 'save', help = 'Guardar las imagenes de salida')
parser.set_defaults(show = True)
parser.set_defaults(save = False)
args = parser.parse_args()

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


path = os.getcwd()
path_model = path + "/singlepose_thunder_quantized.tflite"
#path_model = path + "/singlepose_thunder.tflite"

if args.save:
    result_dir = 'results'
    result_path = os.path.join(path, result_dir)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

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
			cv2.line(
				img=frame, 
				pt1=(int(x1), int(y1)),
				pt2=(int(x2), int(y2)), 
				color=color, 
				thickness=2, 
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

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

times = []

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
	loop(image_resized, output_data,threshold=0.11)
    
	# Obtener las proporciones de redimensionamiento
	scale_x = imW / width
	scale_y = imH / height

	# Reescalar coordenadas (si es necesario)
	image_back_to_original = resize_back(image_resized, (imH, imW))
	image_back_to_original_bgr = cv2.cvtColor(image_back_to_original, cv2.COLOR_RGB2BGR)
	#loop(image,rescaled_output_data,threshold=0.11)

	if args.show:
		cv2.imshow(image_path,image_back_to_original_bgr)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
	if args.save:
		image_name = os.path.basename(image_path)
		image_savepath = os.path.join(path,result_dir,image_name)
		cv2.imwrite(image_savepath, image_back_to_original_bgr)

total_time = sum(times)
times_size = len(times)
average_time = total_time / times_size
average_time = average_time * 1000
print(f"Tiempo de inferencia promedio: {average_time:.2f} ms")
