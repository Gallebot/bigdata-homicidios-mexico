# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession

RUTA_CSV = "file:/workspace/bigdata-homicidios-mexico/data/Defunciones_registradas_mortalidad_general.csv"

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

print("\n=== Primeras 30 columnas ===")
print(df.columns[:30])

print("\n=== Columnas que contienen 'Anio' o 'Mes' ===")
cols_filtro = [c for c in df.columns if ("Anio" in c or "Mes" in c or "anio" in c.lower() or "mes" in c.lower())]
print(cols_filtro[:100])

print("\n=== Primeras 5 filas (solo primeras 10 columnas) ===")
df.select(df.columns[:10]).show(5, truncate=False)

spark.stop()
