"""Local interactive laboratory for visual-channel experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .codec import (
    DecodeError,
    apply_channel,
    decode_palette,
    decode_raw,
    encode_palette,
    encode_raw,
    image_to_data_url,
    metrics,
)


app = FastAPI(title="Visual Channel Lab", version="0.1.0")


def _bsc_rate(bits: int | None, accuracy: float | None) -> float | None:
    if bits is None or accuracy is None:
        return None
    error = min(max(1.0 - accuracy, 0.0), 0.5)
    if error == 0.0:
        return float(bits)
    entropy = -error * math.log2(error) - (1.0 - error) * math.log2(1.0 - error)
    return bits * (1.0 - entropy)


class EncodeRequest(BaseModel):
    text: str = Field(max_length=2_000_000)
    mode: Literal["palette", "raw"] = "palette"
    compress: bool = True
    bits_per_cell: int = Field(4, ge=1, le=4)
    cell_size: int = Field(12, ge=2, le=64)
    repetition: int = Field(1, ge=1, le=9)
    jpeg_quality: int = Field(100, ge=5, le=100)
    scale: float = Field(1.0, ge=0.1, le=1.0)
    blur: float = Field(0.0, ge=0.0, le=8.0)
    noise: float = Field(0.0, ge=0.0, le=64.0)


def _encode(request: EncodeRequest):
    payload = request.text.encode("utf-8")
    repetition = request.repetition if request.repetition % 2 else request.repetition + 1
    if request.mode == "raw":
        encoded = encode_raw(payload, compress=request.compress)
    else:
        encoded = encode_palette(
            payload,
            bits_per_cell=request.bits_per_cell,
            cell_size=request.cell_size,
            repetition=repetition,
            compress=request.compress,
        )
    degraded = apply_channel(
        encoded.image,
        jpeg_quality=request.jpeg_quality,
        scale=request.scale,
        blur=request.blur,
        noise=request.noise,
    )
    return encoded, degraded, repetition


@app.post("/api/encode")
def encode(request: EncodeRequest) -> dict:
    encoded, degraded, repetition = _encode(request)
    try:
        if request.mode == "raw":
            decoded, packet = decode_raw(degraded)
        else:
            decoded, packet = decode_palette(
                degraded,
                bits_per_cell=request.bits_per_cell,
                cell_size=request.cell_size,
                repetition=repetition,
            )
        decoded_text = decoded.decode("utf-8", errors="replace")
        decode_error = None
        exact = decoded == request.text.encode("utf-8")
        crc = f"{packet.crc32:08x}"
    except DecodeError as exc:
        decoded_text = ""
        decode_error = str(exc)
        exact = False
        crc = None
    result_metrics = metrics(encoded)
    result_metrics["payload_characters"] = len(request.text)
    result_metrics["utf8_bytes"] = len(request.text.encode("utf-8"))
    result_metrics["repetition"] = repetition
    return {
        "original_image": image_to_data_url(encoded.image),
        "channel_image": image_to_data_url(degraded),
        "decoded_text": decoded_text,
        "decode_error": decode_error,
        "exact": exact,
        "crc32": crc,
        "metrics": result_metrics,
    }


@app.post("/api/sweep")
def sweep(request: EncodeRequest) -> dict:
    encoded, _, repetition = _encode(request)
    rows = []
    for quality in (100, 95, 90, 80, 70, 60, 50, 35, 20, 10):
        candidate = apply_channel(
            encoded.image,
            jpeg_quality=quality,
            scale=request.scale,
            blur=request.blur,
            noise=request.noise,
        )
        try:
            if request.mode == "raw":
                decoded, _ = decode_raw(candidate)
            else:
                decoded, _ = decode_palette(
                    candidate,
                    bits_per_cell=request.bits_per_cell,
                    cell_size=request.cell_size,
                    repetition=repetition,
                )
            exact = decoded == request.text.encode("utf-8")
            error = None
        except DecodeError as exc:
            exact = False
            error = str(exc)
        rows.append({"jpeg_quality": quality, "exact": exact, "error": error})
    return {"rows": rows, "metrics": metrics(encoded)}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/api/results")
def probe_results() -> dict:
    runs = Path(__file__).resolve().parent.parent / "runs"
    rows = []
    for path in sorted(runs.glob("qwen_probe*.json")) if runs.exists() else []:
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for result in document.get("results", []):
            bits = result.get("bits_per_visual_token")
            bit_accuracy = result.get("bit_accuracy")
            rows.append(
                {
                    "run": path.name,
                    "bits": bits,
                    "feature_view": result.get("feature_view", "final (legacy)"),
                    "bit_accuracy": bit_accuracy,
                    "bsc_rate": result.get("bsc_equivalent_bits_per_token")
                    or _bsc_rate(bits, bit_accuracy),
                    "ci_low": result.get("bit_accuracy_ci95_low"),
                    "ci_high": result.get("bit_accuracy_ci95_high"),
                    "token_exact": result.get("token_exact_accuracy"),
                    "image_exact": result.get("image_exact_accuracy"),
                    "clipping": result.get("clipping_fraction"),
                    "train_rows": result.get("train_rows"),
                }
            )
    llm_rows = []
    for path in sorted(runs.glob("qwen_llm_probe*.json")) if runs.exists() else []:
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for result in document.get("results", []):
            bits = result.get("bits_per_visual_token")
            bit_accuracy = result.get("bit_accuracy")
            llm_rows.append(
                {
                    "run": path.name,
                    "bits": bits,
                    "depth": result.get("depth"),
                    "bit_accuracy": bit_accuracy,
                    "bsc_rate": result.get("bsc_equivalent_bits_per_token")
                    or _bsc_rate(bits, bit_accuracy),
                }
            )
    rendered_rows = []
    rendered_path = runs / "rendered_text.json"
    if rendered_path.exists():
        try:
            document = json.loads(rendered_path.read_text())
        except (OSError, json.JSONDecodeError):
            document = {}
        for result in document.get("results", []):
            rendered_rows.append(
                {
                    "characters": result.get("characters"),
                    "source_bits_per_visual_token": result.get(
                        "source_bits_per_visual_token"
                    ),
                    "character_accuracy": result.get("character_accuracy"),
                    "edit_distance": result.get("edit_distance"),
                    "exact": result.get("exact"),
                }
            )
    return {"rows": rows, "llm_rows": llm_rows, "rendered_rows": rendered_rows}


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual Channel Lab</title>
<style>
:root{color-scheme:dark;--bg:#0b0f14;--panel:#121923;--line:#263244;--ink:#ecf2f8;--muted:#8fa2b7;--hot:#73e6ba;--bad:#ff7d8b;--blue:#7cb7ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#152133 0,var(--bg) 38%);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,sans-serif}
main{max-width:1440px;margin:auto;padding:24px}.top{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:18px}h1{font-size:26px;margin:0}.sub{color:var(--muted);max-width:760px}.grid{display:grid;grid-template-columns:360px 1fr;gap:16px}.panel{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 14px 40px #0005}.controls{display:grid;grid-template-columns:1fr 1fr;gap:12px}.wide{grid-column:1/-1}label{display:block;color:var(--muted);font-size:12px;margin-bottom:5px}textarea,select,input{width:100%;background:#0a1018;border:1px solid var(--line);border-radius:8px;color:var(--ink);padding:9px}textarea{height:190px;resize:vertical}input[type=range]{padding:0}.rangehead{display:flex;justify-content:space-between}button{border:0;border-radius:9px;padding:10px 14px;background:var(--blue);color:#07111e;font-weight:700;cursor:pointer}button.secondary{background:#243244;color:var(--ink)}.buttons{display:flex;gap:8px}.images{display:grid;grid-template-columns:1fr 1fr;gap:12px}.imagebox{min-height:270px;background:#090d12;border:1px solid var(--line);border-radius:10px;display:grid;place-items:center;overflow:auto}.imagebox img{max-width:100%;image-rendering:pixelated}.tag{display:flex;justify-content:space-between;margin-bottom:6px;color:var(--muted)}.result{margin-top:14px;display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.metric{background:#0a1018;border:1px solid var(--line);border-radius:9px;padding:10px}.metric b{display:block;font-size:18px;color:var(--hot)}.status{font-size:18px;font-weight:800}.ok{color:var(--hot)}.bad{color:var(--bad)}pre{white-space:pre-wrap;max-height:170px;overflow:auto;background:#090d12;padding:10px;border-radius:9px;border:1px solid var(--line)}table{width:100%;border-collapse:collapse;margin-top:10px}td,th{text-align:left;border-bottom:1px solid var(--line);padding:7px}.foot{color:var(--muted);margin-top:12px;font-size:12px}@media(max-width:900px){.grid{grid-template-columns:1fr}.result{grid-template-columns:1fr 1fr}.images{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<div class="top"><div><h1>Visual Channel Lab</h1><div class="sub">Pack text into pixels, damage the image, and measure what survives. This is the deterministic control for the learned Qwen channel.</div></div><div id="status" class="status">Ready</div></div>
<div class="grid">
<section class="panel"><div class="controls">
  <div class="wide"><label>Payload</label><textarea id="text">The visual pathway is a noisy communication channel. How many exact bits survive?</textarea></div>
  <div><label>Codec</label><select id="mode"><option value="palette">Robust color cells</option><option value="raw">Raw RGB upper bound</option></select></div>
  <div><label>Compression</label><select id="compress"><option value="true">zlib when smaller</option><option value="false">off</option></select></div>
  <div><label>Bits / color cell</label><select id="bpc"><option>1</option><option>2</option><option>3</option><option selected>4</option></select></div>
  <div><label>Cell size</label><select id="cell"><option>4</option><option>8</option><option selected>12</option><option>16</option><option>24</option></select></div>
  <div><label>Bit repetition</label><select id="rep"><option selected>1</option><option>3</option><option>5</option><option>7</option></select></div>
  <div></div>
  <div class="wide"><div class="rangehead"><label>JPEG quality</label><span id="qv">100</span></div><input id="quality" type="range" min="5" max="100" value="100"></div>
  <div class="wide"><div class="rangehead"><label>Resize scale</label><span id="sv">1.00</span></div><input id="scale" type="range" min="10" max="100" value="100"></div>
  <div class="wide"><div class="rangehead"><label>Gaussian blur</label><span id="bv">0.0</span></div><input id="blur" type="range" min="0" max="40" value="0"></div>
  <div class="wide"><div class="rangehead"><label>Pixel noise σ</label><span id="nv">0</span></div><input id="noise" type="range" min="0" max="40" value="0"></div>
  <div class="wide buttons"><button onclick="run()">Encode + decode</button><button class="secondary" onclick="sweep()">JPEG sweep</button></div>
</div><div class="foot">Raw RGB should fail under almost any transform. Color cells trade density for distance between codewords.</div></section>
<section class="panel">
  <div class="images"><div><div class="tag"><span>Encoded PNG</span><span id="dims"></span></div><div class="imagebox"><img id="original"></div></div><div><div class="tag"><span>After channel</span><span id="channel"></span></div><div class="imagebox"><img id="degraded"></div></div></div>
  <div id="metrics" class="result"></div>
  <h3>Recovered payload</h3><pre id="decoded">—</pre><div id="sweep"></div>
  <div id="probeResults"><h3>Frozen-Qwen results</h3><span class="sub">No completed probe points yet.</span></div>
</section></div></main>
<script>
for(const [id,out,fn] of [['quality','qv',v=>v],['scale','sv',v=>(v/100).toFixed(2)],['blur','bv',v=>(v/5).toFixed(1)],['noise','nv',v=>v]]){document.getElementById(id).oninput=e=>document.getElementById(out).textContent=fn(+e.target.value)}
function payload(){return{text:document.getElementById('text').value,mode:mode.value,compress:compress.value==='true',bits_per_cell:+bpc.value,cell_size:+cell.value,repetition:+rep.value,jpeg_quality:+quality.value,scale:+scale.value/100,blur:+blur.value/5,noise:+noise.value}}
async function request(path){status.textContent='Working…';status.className='status';const r=await fetch(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload())});if(!r.ok)throw new Error(await r.text());return r.json()}
function showMetrics(m){dims.textContent=`${m.width}×${m.height}`;metrics.innerHTML=[['UTF-8 bytes',m.utf8_bytes??m.original_bytes],['Est. visual tokens',m.estimated_visual_tokens],['Bits / visual token',m.source_bits_per_visual_token],['Bits / pixel',m.source_bits_per_pixel]].map(([k,v])=>`<div class="metric"><span>${k}</span><b>${v}</b></div>`).join('')}
async function run(){try{const d=await request('/api/encode');original.src=d.original_image;degraded.src=d.channel_image;showMetrics(d.metrics);decoded.textContent=d.decode_error||d.decoded_text;status.textContent=d.exact?'EXACT + CRC PASS':'DECODE FAILED';status.className='status '+(d.exact?'ok':'bad');channel.textContent=`JPEG ${quality.value}, scale ${(+scale.value/100).toFixed(2)}`;document.getElementById('sweep').innerHTML=''}catch(e){status.textContent='ERROR';status.className='status bad';decoded.textContent=e}}
async function sweep(){try{const d=await request('/api/sweep');showMetrics(d.metrics);document.getElementById('sweep').innerHTML='<h3>JPEG robustness</h3><table><tr><th>Quality</th><th>Exact block recovery</th></tr>'+d.rows.map(r=>`<tr><td>${r.jpeg_quality}</td><td class="${r.exact?'ok':'bad'}">${r.exact?'PASS':'FAIL'}</td></tr>`).join('')+'</table>';status.textContent='SWEEP COMPLETE';status.className='status ok'}catch(e){status.textContent='ERROR';status.className='status bad';decoded.textContent=e}}
async function loadProbeResults(){const d=await fetch('/api/results').then(r=>r.json());let html='';if(d.rows.length)html+='<h3>Frozen-Qwen representation probe</h3><table><tr><th>View</th><th>Load</th><th>Bit accuracy</th><th>BSC-equivalent rate</th><th>Exact token</th><th>Exact image</th><th>Clip</th></tr>'+d.rows.map(r=>`<tr title="${r.run}"><td>${r.feature_view}</td><td>${r.bits} bit/token</td><td>${r.bit_accuracy==null?'—':(100*r.bit_accuracy).toFixed(2)+'%'}</td><td>${r.bsc_rate==null?'—':r.bsc_rate.toFixed(3)+' bit/token'}</td><td>${r.token_exact==null?'—':(100*r.token_exact).toFixed(2)+'%'}</td><td>${r.image_exact==null?'—':(100*r.image_exact).toFixed(2)+'%'}</td><td>${r.clipping==null?'—':(100*r.clipping).toFixed(2)+'%'}</td></tr>`).join('')+'</table>';if(d.llm_rows.length)html+='<h3>Frozen-LLM depth trace</h3><table><tr><th>Depth</th><th>Load</th><th>Bit accuracy</th><th>BSC-equivalent rate</th></tr>'+d.llm_rows.map(r=>`<tr title="${r.run}"><td>${r.depth}</td><td>${r.bits} bit/token</td><td>${(100*r.bit_accuracy).toFixed(2)}%</td><td>${r.bsc_rate.toFixed(3)} bit/token</td></tr>`).join('')+'</table>';if(d.rendered_rows.length)html+='<h3>End-to-end rendered Base32</h3><table><tr><th>Characters</th><th>Source rate</th><th>Character accuracy</th><th>Edit distance</th><th>Exact</th></tr>'+d.rendered_rows.map(r=>`<tr><td>${r.characters}</td><td>${r.source_bits_per_visual_token.toFixed(2)} bit/token</td><td>${(100*r.character_accuracy).toFixed(2)}%</td><td>${r.edit_distance}</td><td class="${r.exact?'ok':'bad'}">${r.exact?'PASS':'FAIL'}</td></tr>`).join('')+'</table>';if(html)probeResults.innerHTML=html}
run();
loadProbeResults();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
