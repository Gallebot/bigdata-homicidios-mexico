# -*- coding: utf-8 -*-
import io, sys
from pyspark.sql import SparkSession

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

IN_CSV = "file:/workspace/bigdata-homicidios-mexico/data/Defunciones_registradas_mortalidad_general.csv"

spark = SparkSession.builder.appName("Homicidios-Preview-Raw").getOrCreate()

df = spark.read.option("header", False).option("inferSchema", False).csv(IN_CSV)

print("rows:", df.count())
print("cols:", len(df.columns))

cols12 = df.columns[:12]
rows = df.select(*cols12).take(15)

print("\n=== Primeras 15 filas (primeras 12 columnas) ===")
for i, r in enumerate(rows):
    print(i, [("" if v is None else str(v)) for v in r])

spark.stop()

