# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

IN_CSV = "file:/workspace/bigdata-homicidios-mexico/data/Defunciones_registradas_mortalidad_general.csv"
OUT_PARQUET = "file:/workspace/bigdata-homicidios-mexico/output/parquet_homicidios_tidy"

spark = SparkSession.builder.appName("Homicidios-Build-Tidy-Long").getOrCreate()
sc = spark.sparkContext


# Helpers de texto 

def to_text(v):
    """Regresa texto seguro (unicode en Py2 / str en Py3) sin romper por ASCII."""
    if v is None:
        try:
            return u""
        except Exception:
            return ""
    try:
        # Python 2
        if isinstance(v, unicode):
            return v
        if isinstance(v, str):  # bytes
            return v.decode("utf-8", "replace")
        return unicode(v)
    except NameError:
        # Python 3
        return str(v)

def clean_text(v):
    s = to_text(v).strip()
    # arreglar basura típica por encoding en el archivo
    try:
        s = s.replace(u"a�o", u"año").replace(u"a�os", u"años")
        s = s.replace(u"A�o", u"Año").replace(u"A�os", u"Años")
    except Exception:
        # por si cae en Py3 y ya es str normal
        s = s.replace("a�o", "año").replace("a�os", "años")
        s = s.replace("A�o", "Año").replace("A�os", "Años")
    return s

def parse_int(x):
    if x is None:
        return 0
    s = to_text(x).strip()
    if s == "" or s == " ":
        return 0
    try:
        return int(float(s))
    except Exception:
        return 0


# Leer raw sin header

df_raw = spark.read.option("header", False).option("inferSchema", False).csv(IN_CSV)

# Convertir a RDD con índice de fila
rdd_idx = df_raw.rdd.zipWithIndex().map(lambda x: (x[1], x[0]))

# Tomar filas de encabezado (0=estado, 2=año, 3=mes)
header = dict(rdd_idx.filter(lambda t: t[0] in (0, 2, 3)).collect())
row_estado = [clean_text(v) for v in header[0]]
row_anio   = [clean_text(v) for v in header[2]]
row_mes    = [clean_text(v) for v in header[3]]

# Columnas de datos (desde la 3ra en adelante)
col_count = len(row_estado)
idx_cols = list(range(2, col_count))

# Mapas broadcast: por índice de columna -> estado/anio/mes
estado_map = {j: row_estado[j] for j in idx_cols}
anio_map   = {j: row_anio[j]   for j in idx_cols}
mes_map    = {j: row_mes[j]    for j in idx_cols}

b_estado = sc.broadcast(estado_map)
b_anio   = sc.broadcast(anio_map)
b_mes    = sc.broadcast(mes_map)

def row_to_records(row):
    # row es pyspark Row con columnas _c0, _c1, _c2...
    sexo = clean_text(row[0])
    edad = clean_text(row[1])

    # saltar filas separadoras/vacías
    if (sexo == "" and edad == "") or (sexo == "" or edad == ""):
        return []

    out = []
    em = b_estado.value
    am = b_anio.value
    mm = b_mes.value

    for j in idx_cols:
        estado = em.get(j, "")
        anio   = am.get(j, "")
        mes    = mm.get(j, "")

        if estado == "" or anio == "" or mes == "":
            continue

        val = parse_int(row[j])

        # Guardar SOLO valores > 0
        if val > 0:
            try:
                out.append((sexo, edad, estado, int(to_text(anio).strip()), mes, val))
            except Exception:
                # si el anio no parsea, lo brincamos
                pass

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

# Si la terminal es ASCII, df_long.show() puede no servir, por lo tanto,
# DEJAR comentado si falla por encoding.
# print("=== Preview tidy ===")
# df_long.show(20, truncate=False)

print("=== Conteo registros tidy (solo >0) ===")
print(df_long.count())

# Guardar Parquet
df_long.write.mode("overwrite").parquet(OUT_PARQUET)
print("Guardado en:", OUT_PARQUET)

spark.stop()

