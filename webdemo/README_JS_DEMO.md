
# CSI4999 IDS Demo (JavaScript Only)

This is a **pure HTML/CSS/JS** demo. No backend needed—works on GitHub Pages or any static host.

## Run locally
Just open `index.html` in a browser, or use any static server:
```bash
# Python 3
python -m http.server 8000
# or Node
npx serve .
```
Then go to http://localhost:8000

## Deploy
- GitHub Pages: put these files in `/docs` or the repo root (Pages settings → deploy from `/docs`).
- Any static host (Netlify, Vercel static, S3, Cloudflare Pages).

## Swap in real model
Edit `static/app.js`. Replace the mock score with your API call:
```js
const res = await fetch('https://your-api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(features)
});
const { proba, label, reasons } = await res.json();
```

## Optional Node server
If you prefer Node/Express and an API, add a minimal server (see `server.js`) and deploy to Render/Railway/Fly.
