# -*- coding: utf-8 -*-
import io, sys, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# --- Intenta importar matplotlib (para graficar) ---
try:
    import matplotlib
    matplotlib.use("Agg")  # backend para guardar PNG sin abrir ventana
    import matplotlib.pyplot as plt
except Exception as e:
    print("\n[ERROR] No pude importar matplotlib para generar gráficas.")
    print("Instálalo dentro del contenedor (una sola vez) con:")
    print("  python3 -m pip install --no-cache-dir numpy matplotlib")
    print("\nDetalle:", str(e))
    sys.exit(1)



# RUTAS (LOCAL DENTRO DEL CONTENEDOR)

BASE = "/workspace/bigdata-homicidios-mexico/output"

CSV_MONTHLY_STATE_DIR      = f"{BASE}/csv_monthly_state"
CSV_MONTHLY_STATE_SEXO_DIR = f"{BASE}/csv_monthly_state_sexo"

OUT_PLOTS_DIR = f"{BASE}/graficas"
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)

# Spark lee local con file:
CSV_MONTHLY_STATE_URI      = f"file:{CSV_MONTHLY_STATE_DIR}"
CSV_MONTHLY_STATE_SEXO_URI = f"file:{CSV_MONTHLY_STATE_SEXO_DIR}"



# SPARK

spark = SparkSession.builder.appName("Homicidios-Visualize-Aggregates").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


def must_exist_dir(path: str):
    if not os.path.exists(path):
        print(f"\n[ERROR] No existe la ruta local: {path}")
        print("Solución: ejecuta primero:")
        print("  03_build_tidy_long.py")
        print("  04_aggregate_monthly.py")
        spark.stop()
        sys.exit(1)


# Verificar que existan carpetas CSV (local)
must_exist_dir(CSV_MONTHLY_STATE_DIR)
must_exist_dir(CSV_MONTHLY_STATE_SEXO_DIR)



# CARGA CSV (Spark)

monthly_state = (
    spark.read.option("header", True).csv(CSV_MONTHLY_STATE_URI)
    .withColumn("anio", F.col("anio").cast("int"))
    .withColumn("mes_num", F.col("mes_num").cast("int"))
    .withColumn("homicidios_total", F.col("homicidios_total").cast("double"))
    .dropna(subset=["estado", "anio", "mes_num", "homicidios_total"])
)

monthly_state_sexo = (
    spark.read.option("header", True).csv(CSV_MONTHLY_STATE_SEXO_URI)
    .withColumn("anio", F.col("anio").cast("int"))
    .withColumn("mes_num", F.col("mes_num").cast("int"))
    .withColumn("homicidios_total", F.col("homicidios_total").cast("double"))
    .dropna(subset=["estado", "anio", "mes_num", "sexo", "homicidios_total"])
)

print("\n=== OK: CSV cargados ===")
print("monthly_state:", monthly_state.count(), "registros")
print("monthly_state_sexo:", monthly_state_sexo.count(), "registros")



# 1) TOP 15 ESTADOS (TOTAL HISTÓRICO)

top_states = (
    monthly_state.groupBy("estado")
    .agg(F.sum("homicidios_total").alias("total"))
    .orderBy(F.desc("total"))
    .limit(15)
    .collect()
)

states = [r["estado"] for r in top_states][::-1]
totals = [float(r["total"]) for r in top_states][::-1]

plt.figure(figsize=(10, 6))
plt.barh(states, totals)
plt.title("Top 15 estados con más homicidios (total histórico)")
plt.xlabel("Homicidios (suma)")
plt.ylabel("Estado")
plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS_DIR, "01_top15_estados_total_historico.png"), dpi=150)
plt.close()



# 2) TENDENCIA NACIONAL POR AÑO (SUMA)

national_by_year = (
    monthly_state.groupBy("anio")
    .agg(F.sum("homicidios_total").alias("total"))
    .orderBy("anio")
    .collect()
)

years = [int(r["anio"]) for r in national_by_year]
year_totals = [float(r["total"]) for r in national_by_year]

plt.figure(figsize=(10, 5))
plt.plot(years, year_totals, marker="o", linewidth=1)
plt.title("Tendencia nacional por año (homicidios totales)")
plt.xlabel("Año")
plt.ylabel("Homicidios (suma)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS_DIR, "02_tendencia_nacional_por_anio.png"), dpi=150)
plt.close()



# 3) ESTACIONALIDAD NACIONAL (SUMA POR MES)

by_month = (
    monthly_state.groupBy("mes_num")
    .agg(F.sum("homicidios_total").alias("total"))
    .orderBy("mes_num")
    .collect()
)

mes_nums = [int(r["mes_num"]) for r in by_month]
mes_totals = [float(r["total"]) for r in by_month]

plt.figure(figsize=(10, 5))
plt.bar(mes_nums, mes_totals)
plt.title("Estacionalidad nacional (homicidios totales por mes)")
plt.xlabel("Mes (1=Enero ... 12=Diciembre)")
plt.ylabel("Homicidios (suma)")
plt.xticks(mes_nums)
plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS_DIR, "03_estacionalidad_nacional_por_mes.png"), dpi=150)
plt.close()



# 4) DISTRIBUCIÓN NACIONAL POR SEXO (TOTAL HISTÓRICO)

sexo_total = (
    monthly_state_sexo.groupBy("sexo")
    .agg(F.sum("homicidios_total").alias("total"))
    .orderBy(F.desc("total"))
    .collect()
)

sexos = [r["sexo"] for r in sexo_total]
sexo_vals = [float(r["total"]) for r in sexo_total]

plt.figure(figsize=(7, 5))
plt.bar(sexos, sexo_vals)
plt.title("Distribución nacional por sexo (total histórico)")
plt.xlabel("Sexo")
plt.ylabel("Homicidios (suma)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS_DIR, "04_total_por_sexo.png"), dpi=150)
plt.close()


print("\n=== Gráficas generadas en ===")
print(f"  {OUT_PLOTS_DIR}")
print(" - 01_top15_estados_total_historico.png")
print(" - 02_tendencia_nacional_por_anio.png")
print(" - 03_estacionalidad_nacional_por_mes.png")
print(" - 04_total_por_sexo.png")

spark.stop()
