# -*- coding: utf-8 -*-
import os
from typing import List, Optional, Dict, Any

import duckdb
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

APP_TITLE = "Homicidios MX - API (Parquet)"
app = FastAPI(title=APP_TITLE)

# CORS para que el frontend pueda llamar la API sin broncas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorio base (montado desde docker-compose)
DATA_DIR = os.getenv("DATA_DIR", "/data")  # mapeado a ../output
TIDY_DIR = os.path.join(DATA_DIR, "parquet_homicidios_tidy")

# DuckDB: se lee cualquier .parquet dentro de la carpeta
PARQUET_GLOB = os.path.join(TIDY_DIR, "*.parquet")

# Columnas esperadas en el tidy:

LABEL_COL = os.getenv("LABEL_COL", "homicidios")


def _get_con():
    # conexión por request (evita problemas de threads)
    con = duckdb.connect(database=":memory:", read_only=False)
    return con

def _table_ref(con):
    # VIEW tidy: normaliza mes y crea mes_num + edad_num si no existen en el parquet
    con.execute(f"""
    CREATE OR REPLACE VIEW tidy AS
    SELECT
      *,
      CASE TRIM(mes)
        WHEN 'Enero' THEN 1
        WHEN 'Febrero' THEN 2
        WHEN 'Marzo' THEN 3
        WHEN 'Abril' THEN 4
        WHEN 'Mayo' THEN 5
        WHEN 'Junio' THEN 6
        WHEN 'Julio' THEN 7
        WHEN 'Agosto' THEN 8
        WHEN 'Septiembre' THEN 9
        WHEN 'Octubre' THEN 10
        WHEN 'Noviembre' THEN 11
        WHEN 'Diciembre' THEN 12
        ELSE NULL
      END AS mes_num,
      TRY_CAST(REGEXP_EXTRACT(TRIM(edad), '([0-9]+)', 1) AS INTEGER) AS edad_num
    FROM read_parquet('{PARQUET_GLOB}');
    """)



def _where_and_params(
    estados: Optional[List[str]],
    sexos: Optional[List[str]],
    anio_min: Optional[int],
    anio_max: Optional[int],
    mes_min: Optional[int],
    mes_max: Optional[int],
    edad_min: Optional[int],
    edad_max: Optional[int],
):
    where = []
    params: List[Any] = []

    if estados:
        where.append("estado IN (" + ",".join(["?"] * len(estados)) + ")")
        params.extend(estados)

    if sexos:
        where.append("sexo IN (" + ",".join(["?"] * len(sexos)) + ")")
        params.extend(sexos)

    if anio_min is not None:
        where.append("anio >= ?")
        params.append(anio_min)
    if anio_max is not None:
        where.append("anio <= ?")
        params.append(anio_max)

    if mes_min is not None:
        where.append("mes_num >= ?")
        params.append(mes_min)
    if mes_max is not None:
        where.append("mes_num <= ?")
        params.append(mes_max)

    if edad_min is not None:
        where.append("edad_num >= ?")
        params.append(edad_min)
    if edad_max is not None:
        where.append("edad_num <= ?")
        params.append(edad_max)

    if where:
        return "WHERE " + " AND ".join(where), params
    return "", params


@app.get("/health")
def health():
    # Verificar que el parquet exista
    if not os.path.isdir(TIDY_DIR):
        return {"ok": False, "error": f"No existe carpeta: {TIDY_DIR}"}

    # ¿hay algún parquet?
    files = [f for f in os.listdir(TIDY_DIR) if f.endswith(".parquet")]
    if not files:
        return {"ok": False, "error": f"No hay .parquet dentro de: {TIDY_DIR}"}

    # Probar lectura simple
    con = _get_con()
    try:
        _table_ref(con)
        cnt = con.execute("SELECT COUNT(*) FROM tidy").fetchone()[0]
        cols = [c[0] for c in con.execute("DESCRIBE tidy").fetchall()]
        return {"ok": True, "rows": cnt, "cols": cols, "label_col": LABEL_COL}
    finally:
        con.close()


@app.get("/filters")
def filters():
    con = _get_con()
    try:
        _table_ref(con)

        estados = [r[0] for r in con.execute("SELECT DISTINCT estado FROM tidy ORDER BY estado").fetchall()]
        sexos = [r[0] for r in con.execute("SELECT DISTINCT sexo FROM tidy ORDER BY sexo").fetchall()]
        years = [r[0] for r in con.execute("SELECT DISTINCT anio FROM tidy ORDER BY anio").fetchall()]
        months = [r[0] for r in con.execute("SELECT DISTINCT mes_num FROM tidy ORDER BY mes_num").fetchall()]
        edad_minmax = con.execute("SELECT MIN(edad_num), MAX(edad_num) FROM tidy").fetchone()



        return {
            "estados": estados,
            "sexos": sexos,
            "years": years,
            "months": months,
            "edad_min": int(edad_minmax[0]) if edad_minmax[0] is not None else None,
            "edad_max": int(edad_minmax[1]) if edad_minmax[1] is not None else None,
        }
    finally:
        con.close()


@app.get("/summary")
def summary(
    estado: Optional[List[str]] = Query(default=None),
    sexo: Optional[List[str]] = Query(default=None),
    anio_min: Optional[int] = None,
    anio_max: Optional[int] = None,
    mes_min: Optional[int] = None,
    mes_max: Optional[int] = None,
    edad_min: Optional[int] = None,
    edad_max: Optional[int] = None,
):
    con = _get_con()
    try:
        _table_ref(con)

        where_sql, params = _where_and_params(estado, sexo, anio_min, anio_max, mes_min, mes_max, edad_min, edad_max)

        q = f"""
        SELECT
          SUM({LABEL_COL}) AS total_homicidios,
          AVG({LABEL_COL}) AS prom_por_registro,
          MIN(anio) AS min_anio,
          MAX(anio) AS max_anio
        FROM tidy
        {where_sql}
        """
        row = con.execute(q, params).fetchone()

        return {
            "total_homicidios": float(row[0] or 0.0),
            "prom_por_registro": float(row[1] or 0.0),
            "min_anio": int(row[2]) if row[2] is not None else None,
            "max_anio": int(row[3]) if row[3] is not None else None,
        }
    finally:
        con.close()


@app.get("/timeseries")
def timeseries(
    estado: Optional[List[str]] = Query(default=None),
    sexo: Optional[List[str]] = Query(default=None),
    anio_min: Optional[int] = None,
    anio_max: Optional[int] = None,
    mes_min: Optional[int] = None,
    mes_max: Optional[int] = None,
    edad_min: Optional[int] = None,
    edad_max: Optional[int] = None,
):
    """
    Serie temporal informativa: total por año-mes (sum).
    """
    con = _get_con()
    try:
        _table_ref(con)
        where_sql, params = _where_and_params(estado, sexo, anio_min, anio_max, mes_min, mes_max, edad_min, edad_max)

        q = f"""
        SELECT anio, mes_num, SUM({LABEL_COL}) AS total
        FROM tidy
        {where_sql}
        GROUP BY anio, mes_num
        ORDER BY anio, mes_num
        """
        rows = con.execute(q, params).fetchall()
        return [{"anio": int(a), "mes_num": int(m), "total": float(t)} for a, m, t in rows]
    finally:
        con.close()


@app.get("/top_states")
def top_states(
    n: int = 10,
    sexo: Optional[List[str]] = Query(default=None),
    anio_min: Optional[int] = None,
    anio_max: Optional[int] = None,
    mes_min: Optional[int] = None,
    mes_max: Optional[int] = None,
    edad_min: Optional[int] = None,
    edad_max: Optional[int] = None,
):
    """
    Barras informativas: top N estados por total.
    """
    con = _get_con()
    try:
        _table_ref(con)
        where_sql, params = _where_and_params(None, sexo, anio_min, anio_max, mes_min, mes_max, edad_min, edad_max)

        q = f"""
        SELECT estado, SUM({LABEL_COL}) AS total
        FROM tidy
        {where_sql}
        GROUP BY estado
        ORDER BY total DESC
        LIMIT ?
        """
        rows = con.execute(q, params + [n]).fetchall()
        return [{"estado": e, "total": float(t)} for e, t in rows]
    finally:
        con.close()


@app.get("/query")
def query_rows(
    estado: Optional[List[str]] = Query(default=None),
    sexo: Optional[List[str]] = Query(default=None),
    anio_min: Optional[int] = None,
    anio_max: Optional[int] = None,
    mes_min: Optional[int] = None,
    mes_max: Optional[int] = None,
    edad_min: Optional[int] = None,
    edad_max: Optional[int] = None,
    limit: int = 200,
):
    """
    Devuelve filas (tabla). Aquí puedes elegir qué columnas mostrar.
    """
    con = _get_con()
    try:
        _table_ref(con)
        where_sql, params = _where_and_params(estado, sexo, anio_min, anio_max, mes_min, mes_max, edad_min, edad_max)

        q = f"""
        SELECT estado, sexo, edad_num, anio, mes, mes_num, {LABEL_COL} AS homicidios
        FROM tidy
        {where_sql}
        ORDER BY anio, mes_num, estado
        LIMIT ?
        """

        rows = con.execute(q, params + [limit]).fetchall()

        out = []
        for r in rows:
            out.append({
                "estado": r[0],
                "sexo": r[1],
                "edad": int(r[2]) if r[2] is not None else None,
                "anio": int(r[3]) if r[3] is not None else None,
                "mes": r[4],
                "mes_num": int(r[5]) if r[5] is not None else None,
                "homicidios": float(r[6] or 0.0),
            })
        return out
    finally:
        con.close()
