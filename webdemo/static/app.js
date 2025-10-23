
/**
 * Pure JS mock IDS model for demo purposes.
 * Replace score() with a call to your real API or model later.
 */
const MockIDSModel = {
  bias: -2.0,
  coeffs: {
    packet_rate: 0.015,
    avg_bytes: 0.002,
    syn_flag: 0.4,
    rst_flag: 0.25,
    dst_port: 0.0005,
  },
  sigmoid(x){ return 1/(1+Math.exp(-x)); },
  score(features){
    let z = this.bias;
    for(const k in features){
      const w = this.coeffs[k] || 0;
      z += w * parseFloat(features[k] || 0);
    }
    return this.sigmoid(z);
  },
  explain(features, proba){
    const reasons = [];
    const pr = parseFloat(features.packet_rate||0);
    const ab = parseFloat(features.avg_bytes||0);
    const syn = parseFloat(features.syn_flag||0);
    const rst = parseFloat(features.rst_flag||0);
    const port = parseInt(features.dst_port||0, 10);

    if (pr > 500) reasons.push("High packet rate (possible scan or flood).");
    if (ab > 1500) reasons.push("Large average payload size.");
    if (syn > 3) reasons.push("Elevated SYN count (scan-like patterns).");
    if (rst > 2) reasons.push("Frequent connection resets (RST).");
    if ([21,22,23,80,443,445,3389].includes(port)) reasons.push(`Well-known target port ${port}.`);
    if (!reasons.length) reasons.push("No strong risk indicators detected; pattern looks typical.");
    reasons.push(`Overall risk score: ${proba.toFixed(2)}`);
    return reasons;
  }
};

function byId(id){ return document.getElementById(id); }

const form = byId('demo-form');
const resultBox = byId('result');
const pill = byId('label-pill');
const scoreEl = byId('score');
const kv = byId('features-table').querySelector('tbody');
const explainList = byId('explain-list');

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const features = {
    packet_rate: byId('packet_rate').value,
    avg_bytes: byId('avg_bytes').value,
    syn_flag: byId('syn_flag').value,
    rst_flag: byId('rst_flag').value,
    dst_port: byId('dst_port').value,
  };

  const proba = MockIDSModel.score(features);
  const label = proba >= 0.5 ? 'malicious' : 'benign';
  const reasons = MockIDSModel.explain(features, proba);

  pill.textContent = label.charAt(0).toUpperCase() + label.slice(1);
  resultBox.classList.remove('hidden');
  resultBox.classList.toggle('bad', label === 'malicious');
  resultBox.classList.toggle('good', label === 'benign');
  scoreEl.textContent = proba.toFixed(2);

  // Populate features table
  kv.innerHTML = '';
  Object.entries(features).forEach(([k,v]) => {
    const tr = document.createElement('tr');
    const th = document.createElement('th'); th.textContent = k;
    const td = document.createElement('td'); td.textContent = v;
    tr.appendChild(th); tr.appendChild(td);
    kv.appendChild(tr);
  });

  // Explanations
  explainList.innerHTML = '';
  reasons.forEach(r => {
    const li = document.createElement('li'); li.textContent = r;
    explainList.appendChild(li);
  });
});
