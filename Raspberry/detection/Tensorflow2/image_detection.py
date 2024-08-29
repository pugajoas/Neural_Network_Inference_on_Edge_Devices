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

# Path para las etiquetas
path_labels = path + "/labels.txt"

#Cargar el archivo de las etiquetas
with open(path_labels, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

#Cargar las imagenes para test
path_images = path + "/imagenes"
images = glob.glob(path_images + '/*.jpg') + glob.glob(path_images + '/*.png') + glob.glob(path_images + '/*.bmp')
images = sorted(images)

loaded_model = tf.saved_model.load(path + "/saved_model_efficientdet_lite0")
infer = loaded_model.signatures['serving_default']
times = []
detections = []
min_conf_threshold = 0.39
    
def preprocess_image(image_path):
	image = cv2.imread(image_path)
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	imH, imW, _ = image.shape
	#image_resized = image_resized / 255.0
	input_data = np.expand_dims(image_rgb, axis=0)
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
    detection_classes =  result['output_2'][0].numpy()
    detection_boxes =  result['output_0'][0].numpy()
    num_detections = int( result['output_3'].numpy()[0])
    detection_scores =  result['output_1'][0].numpy()
    
    for i in range(num_detections):
        if ((detection_scores[i] > min_conf_threshold) and (detection_scores[i] <= 1.0)):
            # Se obtienen las dimensiones de las cajas a dibujar
            ymin = int(detection_boxes[i][0])
            xmin = int(detection_boxes[i][1])
            ymax = int(detection_boxes[i][2])
            xmax = int(detection_boxes[i][3])

            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

			# Dibuja las etiquetas en la imagen
            object_name = labels[int(detection_classes[i])]
            label = '%s: %d%%' % (object_name, int(detection_scores[i]*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
            label_ymin = max(ymin, labelSize[1] + 10) 
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 
            
            detections.append([object_name, detection_scores[i], xmin, ymin, xmax, ymax])
    
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
