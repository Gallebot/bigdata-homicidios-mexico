# -*- coding: utf-8 -*-
import io, sys, os, glob, shutil
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Forzar stdout UTF-8 (para evitar errores con ñ / acentos)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

IN_PARQUET = "file:/workspace/bigdata-homicidios-mexico/output/parquet_homicidios_tidy"

OUT_BASE_LOCAL = "/workspace/bigdata-homicidios-mexico/output"
OUT_BASE_URI   = f"file:{OUT_BASE_LOCAL}"

spark = SparkSession.builder.appName("Homicidios-Aggregate-Monthly").getOrCreate()

df = spark.read.parquet(IN_PARQUET)

# Normalizar mes (quitar espacios)
df = df.withColumn("mes", F.trim(F.col("mes")))

# Orden real de meses para poder ordenar bien después
month_order = F.create_map(
    [F.lit(x) for x in [
        "Enero",1,"Febrero",2,"Marzo",3,"Abril",4,"Mayo",5,"Junio",6,
        "Julio",7,"Agosto",8,"Septiembre",9,"Octubre",10,"Noviembre",11,"Diciembre",12
    ]]
)
df = df.withColumn("mes_num", month_order.getItem(F.col("mes")))

# 1) Total mensual por estado (sumando todo sexo y edades)
monthly_state = (
    df.groupBy("estado","anio","mes","mes_num")
      .agg(F.sum("homicidios").alias("homicidios_total"))
      .orderBy("estado","anio","mes_num")
)

print("=== Preview: monthly_state ===")
monthly_state.show(20, truncate=False)

# 2) Total mensual por estado y sexo
monthly_state_sexo = (
    df.groupBy("estado","anio","mes","mes_num","sexo")
      .agg(F.sum("homicidios").alias("homicidios_total"))
      .orderBy("estado","anio","mes_num","sexo")
)

print("=== Preview: monthly_state_sexo ===")
monthly_state_sexo.show(20, truncate=False)

# Helpers: guardar CSV con nombre fijo
def write_single_csv(df_spark, out_dir_local, final_csv_name):
    """
    Escribe el CSV en una carpeta temporal (Spark genera part-*.csv)
    y luego renombra el part a un archivo fijo final_csv_name.
    """
    tmp_dir_local = out_dir_local + "_tmp"
    tmp_dir_uri = "file:" + tmp_dir_local

    # 1) Escribir CSV (1 solo part)
    df_spark.coalesce(1).write.mode("overwrite").option("header", True).csv(tmp_dir_uri)

    # 2) Encontrar el part-*.csv
    part_files = glob.glob(os.path.join(tmp_dir_local, "part-*.csv"))
    if not part_files:
        raise RuntimeError(f"No encontré part-*.csv en {tmp_dir_local}")
    part_file = part_files[0]

    # 3) Asegurar carpeta final y mover como nombre fijo
    os.makedirs(out_dir_local, exist_ok=True)
    final_path = os.path.join(out_dir_local, final_csv_name)
    shutil.move(part_file, final_path)

    # 4) Limpiar basura temporal (_SUCCESS, CRC, etc.)
    shutil.rmtree(tmp_dir_local, ignore_errors=True)

    return final_path

# Guardar Parquet
PARQUET_STATE_URI = f"{OUT_BASE_URI}/parquet_monthly_state"
PARQUET_SEXO_URI  = f"{OUT_BASE_URI}/parquet_monthly_state_sexo"

monthly_state.write.mode("overwrite").parquet(PARQUET_STATE_URI)
monthly_state_sexo.write.mode("overwrite").parquet(PARQUET_SEXO_URI)

# Guardar CSV (con nombre fijo)
CSV_STATE_DIR_LOCAL = f"{OUT_BASE_LOCAL}/csv_monthly_state"
CSV_SEXO_DIR_LOCAL  = f"{OUT_BASE_LOCAL}/csv_monthly_state_sexo"

csv1 = write_single_csv(monthly_state, CSV_STATE_DIR_LOCAL, "monthly_state.csv")
csv2 = write_single_csv(monthly_state_sexo, CSV_SEXO_DIR_LOCAL, "monthly_state_sexo.csv")

print("\nGuardado Parquet:")
print(f" - {PARQUET_STATE_URI}")
print(f" - {PARQUET_SEXO_URI}")

print("\nGuardado CSV (archivo fijo):")
print(f" - {csv1}")
print(f" - {csv2}")

spark.stop()
