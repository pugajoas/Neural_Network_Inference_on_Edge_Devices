# Para una rápida ejecución se tiene un contenedor en Docker con todo lo necesario para ejecutar estos modelos.

## Requisitos

Antes de comenzar, asegúrate de tener los siguientes recursos instalados en tu Raspberry:

- **Requisito 1**: Tener instalado Docker
- **Requisito 2**: Tener instalado xhost

Sigue estos pasos para instalar los diferentes modelos en tu Raspberry:

1. **Instalar el contenedor de Docker**
	Para este paso tenemos que ejecutar el siguiente comando que descargará la imagen del contenedor:
	```sh
	sudo docker pull pugajoas/tfrasp5:v6.0
	```

2. **Correr el contenedor de Docker**
	Para esto tenermos que ejecutar el siguiente comando una vez descargado la imagen del contenedor.
	```sh
	sudo docker run -it --rm --runtime nvidia --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix" -v <directorio_en_el_host>:<directorio_en_el_contenedor> pugajoas/tfnano:v10.0
	```

	--privileged: Nos sirve para poder utilizar la webcam dentro del contenedor

	--env="DISPLAY" Esto establece la variable DISPLAY dentro del contenedor con el mismo valor que en tu máquina host, permitiendo que el contenedor acceda a la pantalla del host.

	--volume="/tmp/.X11-unix:/tmp/.X11-unix" es el directorio donde el servidor X almacena los archivos de socket que permiten la comunicación entre las aplicaciones gráficas y el servidor X.

	-v <directorio_en_el_host>:<directorio_en_el_contenedor> El <directorio_en_el_host> es donde se encuentra nuestro repositorio de este Github, esto para pasar el repositorio al contenedor de Docker en la ruta específicada.
