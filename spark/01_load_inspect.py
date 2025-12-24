from pyspark.sql import SparkSession

RUTA_CSV = "../data/Defunciones_registradas_mortalidad_general.csv"

spark = SparkSession.builder \
    .appName("Homicidios-Load-Inspect") \
    .getOrCreate()

df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv(RUTA_CSV)

print("=== Filas y columnas ===")
print("rows:", df.count())
print("cols:", len(df.columns))

print("=== Schema ===")
df.printSchema()

print("=== Primeras 5 filas ===")
df.show(5, truncate=False)

spark.stop()
