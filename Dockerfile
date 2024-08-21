# Usar una imagen base oficial de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requisitos y instalar las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar los archivos del proyecto al directorio de trabajo del contenedor
COPY pipeline.py .
COPY dataset/test.csv .
COPY models/modelo_best.pkl models/

# Comando para ejecutar el script principal
CMD ["python", "pipeline.py"]
