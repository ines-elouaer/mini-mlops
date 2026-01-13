const formEl = document.getElementById("form");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const rawEl = document.getElementById("raw");

let expectedFeatures = [];

function setStatus(msg) {
  statusEl.textContent = msg;
}

function cssSafe(s) {
  return s.replaceAll(" ", "_").replaceAll(".", "_").replaceAll("/", "_");
}

function labelFromPrediction(pred) {
  // Si ton dataset suit sklearn: 0=malignant, 1=benign
  // Ici on affiche clair
  return pred === 1 ? "Bénin (1)" : "Malin (0)";
}

function renderForm(features) {
  formEl.innerHTML = "";
  for (const f of features) {
    const wrap = document.createElement("div");
    wrap.innerHTML = `
      <label class="muted">${f}</label>
      <input type="number" step="any" id="f_${cssSafe(f)}" placeholder="valeur..." />
    `;
    formEl.appendChild(wrap);
  }
}

function collectValues() {
  const obj = {};
  for (const f of expectedFeatures) {
    const id = "f_" + cssSafe(f);
    const val = document.getElementById(id).value;
    obj[f] = val === "" ? null : Number(val);
  }
  return obj;
}

function fillValues(values) {
  for (const f of expectedFeatures) {
    const id = "f_" + cssSafe(f);
    if (values[f] !== undefined) {
      document.getElementById(id).value = values[f];
    }
  }
}

async function loadFeatures() {
  try {
    setStatus("Chargement des features...");
    const r = await fetch("/features");
    const data = await r.json();

    if (!data.expected_features) {
      setStatus("❌ Erreur /features: " + JSON.stringify(data));
      return;
    }

    expectedFeatures = data.expected_features;
    renderForm(expectedFeatures);
    setStatus(`✅ ${expectedFeatures.length} features chargées.`);
  } catch (e) {
    setStatus("❌ Erreur réseau: " + e);
  }
}

async function loadExample() {
  if (!expectedFeatures.length) {
    setStatus("⚠️ Clique d'abord sur 'Charger les features'.");
    return;
  }

  const ex = {
    "mean radius": 17.99,
    "mean texture": 10.38,
    "mean perimeter": 122.8,
    "mean area": 1001.0,
    "mean smoothness": 0.1184,
    "mean compactness": 0.2776,
    "mean concavity": 0.3001,
    "mean concave points": 0.1471,
    "mean symmetry": 0.2419,
    "mean fractal dimension": 0.07871,
    "radius error": 1.095,
    "texture error": 0.9053,
    "perimeter error": 8.589,
    "area error": 153.4,
    "smoothness error": 0.006399,
    "compactness error": 0.04904,
    "concavity error": 0.05373,
    "concave points error": 0.01587,
    "symmetry error": 0.03003,
    "fractal dimension error": 0.006193,
    "worst radius": 25.38,
    "worst texture": 17.33,
    "worst perimeter": 184.6,
    "worst area": 2019.0,
    "worst smoothness": 0.1622,
    "worst compactness": 0.6656,
    "worst concavity": 0.7119,
    "worst concave points": 0.2654,
    "worst symmetry": 0.4601,
    "worst fractal dimension": 0.1189
  };

  fillValues(ex);
  setStatus("✅ Exemple rempli.");
}

async function predict() {
  if (!expectedFeatures.length) {
    setStatus("⚠️ Clique d'abord sur 'Charger les features'.");
    return;
  }

  const features = collectValues();
  const missing = Object.entries(features)
    .filter(([_, v]) => v === null || Number.isNaN(v))
    .map(([k]) => k);

  if (missing.length > 0) {
    setStatus(`❌ Il manque ${missing.length} valeurs. Exemple: ${missing[0]}`);
    return;
  }

  try {
    setStatus("Prédiction en cours...");
    const r = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features })
    });

    const data = await r.json();
    rawEl.textContent = JSON.stringify(data, null, 2);

    if (!r.ok) {
      resultEl.innerHTML = `<span class="bad">Erreur</span>`;
      setStatus("❌ " + (data.detail ? data.detail : "Erreur API"));
      return;
    }

    const pred = data.prediction;
    const proba = data.proba;

    resultEl.innerHTML = `
      <span class="ok">✅ Prediction:</span> ${labelFromPrediction(pred)}
      <br/>
      <span class="ok">✅ Proba:</span> ${proba !== null ? proba.toFixed(6) : "N/A"}
      <br/>
      <span class="muted">Features utilisées: ${data.used_features_count}</span>
    `;
    setStatus("✅ OK");
  } catch (e) {
    setStatus("❌ Erreur réseau: " + e);
  }
}

document.getElementById("btnLoad").onclick = loadFeatures;
document.getElementById("btnFillExample").onclick = loadExample;
document.getElementById("btnPredict").onclick = predict;
