let tsChart = null;
let topChart = null;

function $(id){ return document.getElementById(id); }

function apiBase(){
  return $("apiBase").value.replace(/\/+$/, "");
}

function selectedMulti(selectEl){
  return Array.from(selectEl.selectedOptions).map(o => o.value);
}

function qsFromFilters(includeEstado=true){
  const params = new URLSearchParams();

  const estados = selectedMulti($("estados"));
  const sexos   = selectedMulti($("sexos"));

  const anioMin = $("anioMin").value;
  const anioMax = $("anioMax").value;
  const mesMin  = $("mesMin").value;
  const mesMax  = $("mesMax").value;
  const edadMin = $("edadMin").value;
  const edadMax = $("edadMax").value;

  if (includeEstado && estados.length) estados.forEach(v => params.append("estado", v));
  if (sexos.length) sexos.forEach(v => params.append("sexo", v));

  if (anioMin !== "") params.set("anio_min", anioMin);
  if (anioMax !== "") params.set("anio_max", anioMax);

  if (mesMin !== "") params.set("mes_min", mesMin);
  if (mesMax !== "") params.set("mes_max", mesMax);

  if (edadMin !== "") params.set("edad_min", edadMin);
  if (edadMax !== "") params.set("edad_max", edadMax);

  return params;
}

async function getJSON(path){
  const url = `${apiBase()}${path}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function fmtNum(x){
  if (x === null || x === undefined) return "-";
  return Number(x).toLocaleString("es-MX", {maximumFractionDigits: 2});
}

async function ping(){
  $("pingStatus").textContent = "Probando...";
  try{
    const j = await getJSON("/health");
    $("pingStatus").textContent = j.ok ? `OK (${fmtNum(j.rows)} filas)` : `ERROR: ${j.error}`;
  }catch(e){
    $("pingStatus").textContent = `ERROR: ${e.message}`;
  }
}

async function loadFilters(){
  $("status").textContent = "Cargando filtros...";
  try{
    const f = await getJSON("/filters");

    // estados
    $("estados").innerHTML = "";
    f.estados.forEach(e => {
      const opt = document.createElement("option");
      opt.value = e;
      opt.textContent = e;
      $("estados").appendChild(opt);
    });

    // sexos
    $("sexos").innerHTML = "";
    f.sexos.forEach(s => {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s;
      $("sexos").appendChild(opt);
    });

    // rangos default
    $("anioMin").value = f.years[0] ?? "";
    $("anioMax").value = f.years[f.years.length-1] ?? "";
    $("mesMin").value  = 1;
    $("mesMax").value  = 12;

    $("edadMin").value = f.edad_min ?? "";
    $("edadMax").value = f.edad_max ?? "";

    $("status").textContent = "Filtros listos.";
  }catch(e){
    $("status").textContent = `Error: ${e.message}`;
  }
}

async function loadSummary(){
  const params = qsFromFilters(true);
  const path = "/summary?" + params.toString();
  const s = await getJSON(path);

  $("kpiTotal").textContent = fmtNum(s.total_homicidios);
  $("kpiAvg").textContent   = fmtNum(s.prom_por_registro);
  $("kpiYears").textContent = (s.min_anio && s.max_anio) ? `${s.min_anio}–${s.max_anio}` : "-";
}

function upsertTimeSeriesChart(points){
  // points: [{anio, mes_num, total}]
  const labels = points.map(p => `${p.anio}-${String(p.mes_num).padStart(2,"0")}`);
  const data   = points.map(p => p.total);

  const ctx = $("tsChart").getContext("2d");
  if (tsChart) tsChart.destroy();

  tsChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Total homicidios",
        data,
        tension: 0.2,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: true } },
      scales: {
        x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } }
      }
    }
  });
}

async function loadTimeSeries(){
  const params = qsFromFilters(true);
  const path = "/timeseries?" + params.toString();
  const ts = await getJSON(path);
  upsertTimeSeriesChart(ts);
}

function upsertTopChart(rows){
  // rows: [{estado, total}]
  const labels = rows.map(r => r.estado);
  const data   = rows.map(r => r.total);

  const ctx = $("topChart").getContext("2d");
  if (topChart) topChart.destroy();

  topChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Total homicidios",
        data
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: true } }
    }
  });
}

async function loadTopStates(){
  // top_states NO usa estado (porque es el top de estados), pero sí usa el resto de filtros
  const params = qsFromFilters(false);
  params.set("n", $("topN").value || "10");
  const path = "/top_states?" + params.toString();
  const top = await getJSON(path);
  upsertTopChart(top);
}

function renderTable(rows){
  const tbody = $("tbl").querySelector("tbody");
  tbody.innerHTML = "";
  rows.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.estado ?? ""}</td>
      <td>${r.sexo ?? ""}</td>
      <td>${r.edad ?? ""}</td>
      <td>${r.anio ?? ""}</td>
      <td>${r.mes ?? ""}</td>
      <td>${r.mes_num ?? ""}</td>
      <td>${fmtNum(r.homicidios)}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadTable(){
  const params = qsFromFilters(true);
  params.set("limit", $("limit").value || "200");
  const path = "/query?" + params.toString();
  const rows = await getJSON(path);
  renderTable(rows);
}

async function applyAll(){
  $("status").textContent = "Actualizando…";
  try{
    await loadSummary();
    await loadTimeSeries();
    await loadTopStates();
    await loadTable();
    $("status").textContent = "Listo";
  }catch(e){
    $("status").textContent = `Error: ${e.message}`;
  }
}

$("btnPing").addEventListener("click", ping);
$("btnLoadFilters").addEventListener("click", loadFilters);
$("btnApply").addEventListener("click", applyAll);
$("btnTop").addEventListener("click", loadTopStates);

// arranque
(async () => {
  await ping();
  await loadFilters();
  await applyAll();
})();
