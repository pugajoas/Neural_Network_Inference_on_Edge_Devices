import os
import cv2
import numpy as np
import glob
import time
from tensorflow.lite.python.interpreter import Interpreter

path = os.getcwd()
path_model = path + "/mobilenetv2quantized.tflite"
#path_model = path + "/mobilenetv2.tflite"

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
    predicted_class = np.argmax(output_data)
    if floating_model:
        percentage = output_data[0][predicted_class] * 100
    else:
        percentage = (output_data[0][predicted_class] / 255) * 100
        
    print(f"Predicción: {labels[predicted_class]} con {percentage:.2f}%")

total_time = sum(times)
times_size = len(times)
average_time = total_time / times_size
average_time = average_time * 1000
print(f"Tiempo de inferencia promedio: {average_time:.2f} ms")
