# -*- coding: utf-8 -*-
import sys
import codecs
from pyspark.sql import SparkSession

IN_CSV = "file:/workspace/bigdata-homicidios-mexico/data/Defunciones_registradas_mortalidad_general.csv"

# --- Soporte PY2/PY3 para texto ---
PY2 = (sys.version_info[0] == 2)
try:
    text_type = unicode  # PY2
except NameError:
    text_type = str      # PY3

# En PY2, envolver stdout en UTF-8 (no existe sys.stdout.buffer)
if PY2:
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout)

def safe_text(v):
    """Convierte cualquier valor a texto imprimible sin romper por encoding."""
    if v is None:
        return ""
    s = text_type(v)  # NO uses str(v) en PY2
    if PY2:
        return s.encode("utf-8", "replace")  # bytes imprimibles
    return s.encode("utf-8", "replace").decode("utf-8")  # str en PY3

spark = SparkSession.builder.appName("Homicidios-Preview-Raw").getOrCreate()

df = spark.read.option("header", False).option("inferSchema", False).csv(IN_CSV)

print("rows:", df.count())
print("cols:", len(df.columns))

cols12 = df.columns[:12]
rows = df.select(*cols12).take(15)

print("\n=== Primeras 15 filas (primeras 12 columnas) ===")
for i, row in enumerate(rows):
    safe = [safe_text(v) for v in row[:12]]
    print(i, safe)

spark.stop()

