# Análisis de Homicidios en México (Big Data)

Este proyecto realiza un procesamiento de datos masivos utilizando **Apache Spark** sobre un entorno contenedorizado con **Docker**. Permite la limpieza, agregación, predicción y visualización de datos estadísticos.

---

## Requisitos Previos

> [!IMPORTANT]
> Este proyecto está diseñado para ejecutarse exclusivamente en **Ubuntu Linux** (Recomendado 22.04 LTS o superior) debido a las dependencias de Docker y la configuración del Kernel necesaria para Spark.

* **Sistema Operativo:** Ubuntu 22.04 LTS o superior.
* **Memoria RAM:** Mínimo 8GB (Recomendado 16GB).
* **Docker** y **Docker Compose** instalados.

---
Es necesario realizar en orden todos los pasos que se mencionan a continuación para una ejecución correcta


## Fase A: Preparación del Host (dentro de la terminal de Ubuntu)

### 1) Clonar el Proyecto
cd ~/Downloads
git clone https://github.com/Gallebot/bigdata-homicidios-mexico.git

### 2) Instalar Docker (si no estaba instalado)
sudo apt-get update
sudo apt-get install -y git ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu noble stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl enable --now docker

### 3) Permisos Docker 
sudo groupadd -f docker
sudo usermod -aG docker "nombre de usuario"

Nota: Es necesario cerrar sesión y volver a entrar para que los cambios surtan efecto.

### 4) Verificación
docker --version
docker compose version
docker ps
docker pull suhothayan/hadoop-spark-pig-hive:2.9.2


---



## Fase B: Preparación del Host (aún en la terminal de Ubuntu)


### 5) Entrar al contenedor con el repo montado (SIN --rm para no perder instalaciones)
docker run -it \
  --name spark_lab_ws \
  -v ~/Downloads/bigdata-homicidios-mexico:/workspace/bigdata-homicidios-mexico \
  suhothayan/hadoop-spark-pig-hive:2.9.2 \
  bash


## -- A PARTIR DE AQUÍ YA ES LA TERMINAL ROOT --

### 6) Ir a la raíz del repositorio
cd /workspace/bigdata-homicidios-mexico


### 7) Instalar dependencias dentro del contenedor
apt-get update

apt-get install -y \
  python3 \
  python3-pip \
  python3-setuptools \
  python3-wheel \
  python3-dev \
  build-essential \
  zlib1g-dev \
  libjpeg-dev \
  libpng-dev \
  libfreetype6-dev \
  pkg-config


### 8) Instalación de librerías 
python3 -m pip install --no-cache-dir --upgrade pip
python3 -m pip install --no-cache-dir numpy matplotlib

(Opcional) comprobar que matplotlib quedó instalado en python3
python3 -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"


### 9) Variables de entorno (Spark + UTF-8)  + Forzar Python 3 en Spark

export PYSPARK_PYTHON=/usr/bin/python3
export PYSPARK_DRIVER_PYTHON=/usr/bin/python3
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

for f in \
  spark/01_load_inspect.py \
  spark/02_preview_raw.py \
  spark/03_build_tidy_long.py \
  spark/04_aggregate_monthly.py \
  spark/05_visualize_aggregates.py \
  spark/06_predict_monthly.py
do
  echo "======================================="
  echo "EJECUTANDO: $f"

  PYSPARK_PYTHON=/usr/bin/python3 \
  PYSPARK_DRIVER_PYTHON=/usr/bin/python3 \
  PYTHONIOENCODING=utf-8 \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  spark-submit \
    --conf spark.pyspark.python=/usr/bin/python3 \
    --conf spark.pyspark.driver.python=/usr/bin/python3 \
    "$f" || exit 1
done

---



## Fase C: Levantar el servidor para ver la página (en la terminal de Ubuntu)

Es necesario abrir una terminal nueva para ejecutar estos comandos

### 1) Dirigirse a la ruta correcta

cd Downloads/bigdata-homicidios-mexico

cd web

### 2) Iniciar el docker
docker-compose up --build

O puede quitarse ejecutarse dependiendo la version de docker instalada:
docker compose up --build 

## -- Es obligatorio abrir una terminal nueva y ejecutar lo siguiente --

### 3) Verificar el estado del servidor
curl http://localhost:8000/health

### 4) Dirigirse a la ruta de la página
cd Downloads/bigdata-homicidios-mexico/web/frontend

### 5) Iniciar el servidor
python3 -m http.server 5173

### 6) Dirigirse a la página 
http://localhost:5173/
