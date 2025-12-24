# -*- coding: utf-8 -*-
import io, sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# stdout en UTF-8 aunque la terminal sea ASCII
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

IN_CSV = "file:/workspace/bigdata-homicidios-mexico/data/Defunciones_registradas_mortalidad_general.csv"
OUT_PARQUET = "file:/workspace/bigdata-homicidios-mexico/output/parquet_homicidios_tidy"

spark = SparkSession.builder.appName("Homicidios-Build-Tidy-Long").getOrCreate()
sc = spark.sparkContext

# Leer raw sin header
df_raw = spark.read.option("header", False).option("inferSchema", False).csv(IN_CSV)

# Convertir a RDD con índice de fila
rdd_idx = df_raw.rdd.zipWithIndex().map(lambda x: (x[1], x[0]))

# Tomar filas de encabezado (0=estado, 2=año, 3=mes)
header = dict(rdd_idx.filter(lambda t: t[0] in (0, 2, 3)).collect())
row_estado = [("" if v is None else str(v)).strip() for v in header[0]]
row_anio   = [("" if v is None else str(v)).strip() for v in header[2]]
row_mes    = [("" if v is None else str(v)).strip() for v in header[3]]

# Limpiar estado/mes (quitar espacios)
# Nota: en tus datos, las columnas 0 y 1 son sexo/edad, los valores empiezan en col 2
col_count = len(row_estado)
idx_cols = list(range(2, col_count))

# Mapas broadcast: por índice de columna -> estado/anio/mes
estado_map = {j: row_estado[j] for j in idx_cols}
anio_map   = {j: row_anio[j]   for j in idx_cols}
mes_map    = {j: row_mes[j]    for j in idx_cols}

b_estado = sc.broadcast(estado_map)
b_anio   = sc.broadcast(anio_map)
b_mes    = sc.broadcast(mes_map)

def clean_text(s):
    if s is None:
        return ""
    s = str(s).strip()
    # arreglar el "a�o" típico por encoding
    s = s.replace("a�o", "año").replace("a�os", "años")
    s = s.replace("A�o", "Año").replace("A�os", "Años")
    return s

def parse_int(x):
    if x is None:
        return 0
    s = str(x).strip()
    if s == "" or s == " ":
        return 0
    # algunos CSV traen espacios en vez de vacío
    try:
        return int(float(s))
    except:
        return 0

def row_to_records(row):
    # row es pyspark Row con columnas _c0, _c1, _c2...
    sexo = clean_text(row[0])
    edad = clean_text(row[1])

    # saltar filas separadoras/vacías
    if sexo == "" and edad == "":
        return []

    # si por alguna razón no hay sexo/edad, también saltar
    if sexo == "" or edad == "":
        return []

    out = []
    em = b_estado.value
    am = b_anio.value
    mm = b_mes.value

    # recorrer columnas de valores
    for j in idx_cols:
        estado = em.get(j, "")
        anio   = am.get(j, "")
        mes    = mm.get(j, "")

        # si algo de header está vacío, no generamos registro
        if estado == "" or anio == "" or mes == "":
            continue

        val = parse_int(row[j])

        # Para ahorrar espacio: guardamos SOLO valores > 0 (los vacíos equivalen a 0)
        if val > 0:
            out.append((sexo, edad, estado, int(anio), mes, val))

    return out

# Filtrar solo filas de datos (índice >= 4) y convertir a registros largos
data_rdd = rdd_idx.filter(lambda t: t[0] >= 4).map(lambda t: t[1])
long_rdd = data_rdd.flatMap(row_to_records)

schema = StructType([
    StructField("sexo", StringType(), True),
    StructField("edad", StringType(), True),
    StructField("estado", StringType(), True),
    StructField("anio", IntegerType(), True),
    StructField("mes", StringType(), True),
    StructField("homicidios", IntegerType(), True),
])

df_long = spark.createDataFrame(long_rdd, schema=schema)

print("=== Preview tidy ===")
df_long.show(20, truncate=False)

print("=== Conteo registros tidy (solo >0) ===")
print(df_long.count())

# Guardar Parquet
df_long.write.mode("overwrite").parquet(OUT_PARQUET)
print("Guardado en:", OUT_PARQUET)

spark.stop()

