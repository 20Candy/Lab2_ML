# Define el nombre del contenedor
$containerName = "mi-aplicacion"

# Corre el contenedor
docker run --name $containerName -d -p 4000:5000 mi-aplicacion

# Espera a que el contenedor termine su ejecución
docker wait $containerName

# Copia el archivo de predicciones
docker cp "${containerName}:/app/dataset/predictions.csv" "C:\Users\carev\OneDrive\Documentos\GitHub\Lab2_ML\dataset"

# Opcional: Detener y remover el contenedor
docker stop $containerName
docker rm $containerName

# Informar que el proceso ha terminado
Write-Output "Archivo de predicciones copiado exitosamente"
