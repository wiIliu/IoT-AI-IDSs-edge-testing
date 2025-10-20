
// Optional Node/Express server (only if you want one).
// Run: npm i express cors body-parser && node server.js
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');

const app = express();
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(__dirname));

// Mock predict endpoint. Replace with your real model logic later.
app.post('/api/predict', (req, res) => {
  const f = req.body || {};
  // Simple copy of the JS mock model math
  const bias = -2.0;
  const coeffs = { packet_rate:0.015, avg_bytes:0.002, syn_flag:0.4, rst_flag:0.25, dst_port:0.0005 };
  let z = bias;
  Object.keys(coeffs).forEach(k => { z += (coeffs[k] * parseFloat(f[k]||0)); });
  const proba = 1/(1+Math.exp(-z));
  const label = proba >= 0.5 ? 'malicious' : 'benign';
  const reasons = [];
  const pr = parseFloat(f.packet_rate||0);
  const ab = parseFloat(f.avg_bytes||0);
  const syn = parseFloat(f.syn_flag||0);
  const rst = parseFloat(f.rst_flag||0);
  const port = parseInt(f.dst_port||0, 10);
  if (pr > 500) reasons.push("High packet rate (possible scan or flood).");
  if (ab > 1500) reasons.push("Large average payload size.");
  if (syn > 3) reasons.push("Elevated SYN count (scan-like patterns).");
  if (rst > 2) reasons.push("Frequent connection resets (RST).");
  if ([21,22,23,80,443,445,3389].includes(port)) reasons.push(`Well-known target port ${port}.`);
  if (!reasons.length) reasons.push("No strong risk indicators detected; pattern looks typical.");
  reasons.push(`Overall risk score: ${proba.toFixed(2)}`);
  res.json({ proba, label, reasons });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Serving on http://localhost:${PORT}`));
