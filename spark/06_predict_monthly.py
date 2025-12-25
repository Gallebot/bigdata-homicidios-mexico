# -*- coding: utf-8 -*-
import io, sys, os, shutil, glob
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# RUTAS (TODO LOCAL)

IN_PARQUET_LOCAL = "/workspace/bigdata-homicidios-mexico/output/parquet_monthly_state"
OUT_BASE         = "/workspace/bigdata-homicidios-mexico/output"

PARQUET_PRED_LOCAL_DIR = f"{OUT_BASE}/parquet_predictions_monthly_state"
CSV_PRED_LOCAL_DIR     = f"{OUT_BASE}/csv_predictions_monthly_state"   
CSV_PRED_LOCAL_FILE    = f"{CSV_PRED_LOCAL_DIR}/predictions.csv"       

PLOTS_DIR = f"{OUT_BASE}/graficas_predicciones"


# SPARK

spark = (
    SparkSession.builder
    .appName("Homicidios-Predict-Monthly")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")


# UTILIDADES

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def rm_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)

def write_single_csv_fixed_name_local(df, out_dir_local: str, fixed_csv_path_local: str):
    """
    Escribe df en out_dir_local (Spark crea part-*.csv) y lo renombra a fixed_csv_path_local,
    dejando SOLO predictions.csv (borra _SUCCESS y parts).
    """
    rm_dir(out_dir_local)
    ensure_dir(out_dir_local)

    # Forzar escritura LOCAL con file:
    df.coalesce(1).write.mode("overwrite").option("header", True).csv("file:" + out_dir_local)

    part_files = glob.glob(os.path.join(out_dir_local, "part-*.csv"))
    if not part_files:
        raise RuntimeError(f"No se encontró part-*.csv dentro de: {out_dir_local}")

    part_csv = part_files[0]

    # Renombrar a nombre fijo
    if os.path.exists(fixed_csv_path_local):
        os.remove(fixed_csv_path_local)
    os.rename(part_csv, fixed_csv_path_local)

    # Borrar basura restante
    for name in os.listdir(out_dir_local):
        full = os.path.join(out_dir_local, name)
        if full != fixed_csv_path_local:
            try:
                if os.path.isdir(full):
                    shutil.rmtree(full)
                else:
                    os.remove(full)
            except:
                pass


def save_prediction_plots(pred_df, plots_dir: str, max_year: int):
    """
    Gráficas informativas usando SOLO las predicciones:
    1) Nacional (test): Real vs Predicho por mes
    2) Top 15 estados con mayor MAE en test
    3) Scatter Real vs Predicho (test)
    """
    ensure_dir(plots_dir)

    # Nos quedamos con columnas necesarias y calculamos error absoluto
    p = (pred_df
         .select("estado", "anio", "mes_num", "homicidios_total", "prediccion")
         .withColumn("abs_error", F.abs(F.col("homicidios_total") - F.col("prediccion")))
    )

    # ========== 1) Nacional: Real vs Predicho por mes (año test) ==========
    nat = (p.filter(F.col("anio") == F.lit(max_year))
           .groupBy("mes_num")
           .agg(
               F.sum("homicidios_total").alias("real_total"),
               F.sum("prediccion").alias("pred_total")
           )
           .orderBy("mes_num")
           .collect()
    )

    if nat:
        meses = [int(r["mes_num"]) for r in nat]
        real = [float(r["real_total"]) for r in nat]
        pred = [float(r["pred_total"]) for r in nat]

        plt.figure(figsize=(10, 5))
        plt.plot(meses, real, marker="o", linewidth=1, label="Real")
        plt.plot(meses, pred, marker="o", linewidth=1, label="Predicho")
        plt.title(f"Nacional (Año {max_year}): Real vs Predicho por mes")
        plt.xlabel("Mes (1=Enero ... 12=Diciembre)")
        plt.ylabel("Homicidios (suma)")
        plt.xticks(meses)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"01_real_vs_pred_nacional_{max_year}.png"), dpi=150)
        plt.close()

    # ========== 2) Top estados con mayor MAE (test) ==========
    top_err = (p.filter(F.col("anio") == F.lit(max_year))
               .groupBy("estado")
               .agg(F.avg("abs_error").alias("mae_estado"))
               .orderBy(F.desc("mae_estado"))
               .limit(15)
               .collect()
    )

    if top_err:
        estados = [r["estado"] for r in top_err][::-1]
        maes = [float(r["mae_estado"]) for r in top_err][::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(estados, maes)
        plt.title(f"Top 15 estados con mayor error (MAE) - Año {max_year}")
        plt.xlabel("MAE promedio (|Real - Predicho|)")
        plt.ylabel("Estado")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"02_top15_mae_por_estado_{max_year}.png"), dpi=150)
        plt.close()

    # ========== 3) Scatter Real vs Predicho (test) ==========
    # Tomamos una muestra para no explotar memoria si hay muchos registros
    sample = (p.filter(F.col("anio") == F.lit(max_year))
              .select("homicidios_total", "prediccion")
              .orderBy(F.rand(seed=42))
              .limit(4000)
              .collect()
    )

    if sample:
        x = [float(r["homicidios_total"]) for r in sample]
        y = [float(r["prediccion"]) for r in sample]

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, s=8, alpha=0.4)
        plt.title(f"Real vs Predicho (muestra) - Año {max_year}")
        plt.xlabel("Real")
        plt.ylabel("Predicho")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"03_scatter_real_vs_pred_{max_year}.png"), dpi=150)
        plt.close()

# CHECK INPUT

if not os.path.exists(IN_PARQUET_LOCAL):
    print("\n[ERROR] No existe el parquet de entrada (LOCAL):")
    print(f"  {IN_PARQUET_LOCAL}")
    print("\nSolución:")
    print("  1) Ejecuta 03_build_tidy_long.py (genera parquet_homicidios_tidy)")
    print("  2) Ejecuta 04_aggregate_monthly.py (genera parquet_monthly_state)")
    spark.stop()
    sys.exit(1)


# CARGA

df = spark.read.parquet("file:" + IN_PARQUET_LOCAL)

# Limpieza mínima
df = (df
      .withColumn("estado", F.trim(F.col("estado")))
      .withColumn("mes", F.trim(F.col("mes")))
)

# Asegurar tipos
df = (df
      .withColumn("anio", F.col("anio").cast("int"))
      .withColumn("mes_num", F.col("mes_num").cast("int"))
      .withColumn("homicidios_total", F.col("homicidios_total").cast("double"))
      .dropna(subset=["estado", "anio", "mes_num", "homicidios_total"])
)

print("=== Preview input (monthly_state) ===")
df.orderBy("estado", "anio", "mes_num").show(10, truncate=False)

print("=== Rango de años ===")
df.agg(F.min("anio").alias("min_anio"), F.max("anio").alias("max_anio")).show()


# SPLIT (time-aware)

max_year = df.agg(F.max("anio").alias("max_anio")).collect()[0]["max_anio"]
train_df = df.filter(F.col("anio") < F.lit(max_year))
test_df  = df.filter(F.col("anio") == F.lit(max_year))

# Fallback si queda vacío
train_n = train_df.count()
test_n  = test_df.count()
if train_n == 0 or test_n == 0:
    print("[WARN] Split por año dejó train o test vacío. Usando randomSplit 80/20.")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    train_n = train_df.count()
    test_n  = test_df.count()

print(f"=== Split ===\nTrain: {train_n} registros\nTest:  {test_n} registros\n")


# PIPELINE (features + modelo)

# estado -> index
estado_indexer = StringIndexer(inputCol="estado", outputCol="estado_idx", handleInvalid="keep")

# mes_num -> index (categórico)
mes_indexer = StringIndexer(inputCol="mes_num", outputCol="mes_idx", handleInvalid="keep")

# OneHot
ohe = OneHotEncoderEstimator(
    inputCols=["estado_idx", "mes_idx"],
    outputCols=["estado_ohe", "mes_ohe"],
    handleInvalid="keep"
)

# Features: estado_ohe + mes_ohe + anio
assembler = VectorAssembler(
    inputCols=["estado_ohe", "mes_ohe", "anio"],
    outputCol="features"
)

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="homicidios_total",
    predictionCol="prediccion",
    numTrees=100,
    maxDepth=10,
    seed=42
)

pipeline = Pipeline(stages=[estado_indexer, mes_indexer, ohe, assembler, rf])

model = pipeline.fit(train_df)
pred  = model.transform(test_df)


# MÉTRICAS

e_rmse = RegressionEvaluator(labelCol="homicidios_total", predictionCol="prediccion", metricName="rmse")
e_mae  = RegressionEvaluator(labelCol="homicidios_total", predictionCol="prediccion", metricName="mae")
e_r2   = RegressionEvaluator(labelCol="homicidios_total", predictionCol="prediccion", metricName="r2")

rmse = e_rmse.evaluate(pred)
mae  = e_mae.evaluate(pred)
r2   = e_r2.evaluate(pred)

print("=== Métricas en TEST ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R^2 : {r2:.4f}")

print("\n=== Preview predicciones (test) ===")
(pred
 .select("estado","anio","mes","mes_num","homicidios_total","prediccion")
 .orderBy("estado","anio","mes_num")
 .show(20, truncate=False)
)

# GRÁFICAS INFORMATIVAS (PREDICCIONES)
save_prediction_plots(pred, PLOTS_DIR, max_year)

print("\nGráficas de predicciones guardadas en:")
print(f" - {PLOTS_DIR}")
print(f" - 01_real_vs_pred_nacional_{max_year}.png")
print(f" - 02_top15_mae_por_estado_{max_year}.png")
print(f" - 03_scatter_real_vs_pred_{max_year}.png")



# GUARDADO (LOCAL)

ensure_dir(OUT_BASE)

# 1) Parquet (local)
rm_dir(PARQUET_PRED_LOCAL_DIR)
(pred
 .select("estado","anio","mes","mes_num","homicidios_total","prediccion")
 .write.mode("overwrite")
 .parquet("file:" + PARQUET_PRED_LOCAL_DIR)
)

# 2) CSV (local)
pred_out = (pred
    .select("estado","anio","mes","mes_num","homicidios_total","prediccion")
    .orderBy("estado","anio","mes_num")
)

write_single_csv_fixed_name_local(pred_out, CSV_PRED_LOCAL_DIR, CSV_PRED_LOCAL_FILE)

print("\nGuardado Parquet predicciones (LOCAL):")
print(f" - {PARQUET_PRED_LOCAL_DIR}")

print("\nGuardado CSV predicciones (LOCAL, nombre fijo):")
print(f" - {CSV_PRED_LOCAL_FILE}")

spark.stop()
