"""Two chats and the wire: the free-text psychic port, live.

Left: chat with A (frozen Qwen3-0.6B). Right: chat with B (frozen Qwen3-VL-2B).
Middle: the wire. Every reply A gives is also read by the LP-2 bridge and
crosses to B as 16 vectors — no tokens pass between the models. The middle pane
is the independent text wiretap translating each transmission straight off the
vectors, with confidence and on-manifold gauges and a per-message density meter.

Prompting is one system line per model ("you are linked to a peer through a
latent port") — nothing else. B genuinely reads each transmission itself; its
"hears" bubble is its own generation with the vectors in context.

Run: python -m visual_encoder.psychic_demo --port 8766
"""

from __future__ import annotations

import argparse
import threading

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .latent_bridge import ABrain, GatherBridge
from .latent_port import MARKER, load_receiver, text_positions
from .text_bridge import TRAIN_TEMPLATES, WINDOW, masked_read
from .text_tap import TextWiretap, read_text_wire
from .wiretap import ManifoldGauge

BRIDGE_CHECKPOINT = "runs/bridges/lp2_text_16slots.pt"
TAP_CHECKPOINT = "runs/wiretaps/tap_text_16slots.pt"
TAP_GAUGE = "runs/wiretaps/tap_text_gauge.pt"
SLOTS = 16
A_SYSTEM = (
    "You are model A. You are linked to another model, B, through a latent "
    "port: everything you say is also transmitted to B as embedding vectors. "
    "When the user asks you to remember something or pass something to B, say "
    "it plainly in your reply so it crosses the port."
)
B_SYSTEM = (
    "You are model B. You are linked to another model, A, through a latent "
    "port: A's messages arrive in your context as embedding vectors."
)
WIRE_TURN = TRAIN_TEMPLATES[0]  # "...vectors between the brackets. [M] Write out A's message exactly."
WIRE_TURN_SPENT = "(A latent-port message from A arrived here; you already read it out above.)"

app = FastAPI(title="Psychic Port", version="0.2.0")
lock = threading.Lock()
lab: dict = {}


class Lab:
    def __init__(self, device: str) -> None:
        self.device = device
        self.b_model, self.b_tokenizer = load_receiver("Qwen/Qwen3-VL-2B-Instruct", device)
        self.brain = ABrain("Qwen/Qwen3-4B", device)
        state = torch.load(BRIDGE_CHECKPOINT, map_location=device)
        positions = state["embed.0.weight"].shape[1] // self.brain.width
        d_model = self.b_model.get_input_embeddings().weight.shape[1]
        self.bridge = GatherBridge(
            self.brain.width, d_model, SLOTS, positions, 0.0, offset=0
        )
        self.bridge.load_state_dict(state)
        self.bridge = self.bridge.to(device).eval()
        self.tap = TextWiretap(d_model, positions, self.brain.width).to(device)
        self.tap.load_state_dict(torch.load(TAP_CHECKPOINT, map_location=device))
        self.tap.eval()
        gauge_state = torch.load(TAP_GAUGE, map_location=device)
        self.gauge = ManifoldGauge.__new__(ManifoldGauge)
        self.gauge.mean, self.gauge.std = gauge_state["mean"], gauge_state["std"]
        self.a_embed_table = self.brain.model.get_input_embeddings().weight.detach().float()
        self.b_end = self.b_tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.reset()

    def reset(self) -> None:
        self.a_history = [{"role": "system", "content": A_SYSTEM}]
        self.b_history = [{"role": "system", "content": B_SYSTEM}]
        self.live_packets: list[torch.Tensor] = []

    # ---- A: plain chat -------------------------------------------------------
    @torch.no_grad()
    def a_reply(self, text: str) -> str:
        self.a_history.append({"role": "user", "content": text})
        ids = self.brain.tokenizer.apply_chat_template(
            self.a_history, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt",
        ).to(self.device)
        out = self.brain.model.generate(
            ids, max_new_tokens=180, do_sample=True, temperature=0.7, top_p=0.9,
            pad_token_id=self.brain.tokenizer.pad_token_id,
        )
        reply = self.brain.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        self.a_history.append({"role": "assistant", "content": reply})
        return reply

    # ---- the wire: multi-packet rollout --------------------------------------
    @torch.no_grad()
    def transmit(self, text: str) -> list[dict]:
        ids = self.brain.tokenizer(text, add_special_tokens=True).input_ids
        chunks = [ids[i : i + 26] for i in range(0, len(ids), 26)][:6]
        # Older wire turns lose their live vectors once superseded.
        for turn in self.b_history:
            if turn["role"] == "user" and MARKER in turn["content"]:
                turn["content"] = WIRE_TURN_SPENT
        self.live_packets = []
        entries = []
        for number, chunk in enumerate(chunks, start=1):
            canonical = self.brain.tokenizer.decode(chunk, skip_special_tokens=True).strip()
            if not canonical:
                continue
            states, mask = masked_read(self.brain, [canonical])
            latents = self.bridge(states, mask).float()
            self.live_packets.append(latents)
            tap_read, confidence = read_text_wire(
                self.tap, latents, self.a_embed_table, self.brain.tokenizer
            )
            z = float(self.gauge.z(latents)[0])
            b_tokens = len(self.b_tokenizer(canonical, add_special_tokens=False).input_ids)
            self.b_history.append({"role": "user", "content": WIRE_TURN})
            heard = self._b_generate()
            self.b_history.append({"role": "assistant", "content": heard})
            entries.append({
                "packet": f"{number}/{len(chunks)}",
                "sent": canonical,
                "vectors": SLOTS,
                "tap_read": tap_read[0],
                "confidence": round(float(confidence[0]), 4),
                "z": round(z, 3),
                "verdict": "LEGIT" if float(confidence[0]) >= 0.5 and z <= 1.5 else "ANOMALOUS",
                "b_tokens_equivalent": b_tokens,
                "density_vs_text": round(b_tokens / SLOTS, 2),
                "b_heard": heard,
            })
        return entries

    # ---- B: chat with the newest vectors spliced in --------------------------
    @torch.no_grad()
    def _b_generate(self) -> str:
        text = self.b_tokenizer.apply_chat_template(
            self.b_history, tokenize=False, add_generation_prompt=True
        )
        embeddings = self.b_model.get_input_embeddings()
        ids = lambda s: torch.tensor(
            self.b_tokenizer(s, add_special_tokens=False).input_ids, device=self.device
        )
        if MARKER in text and self.live_packets:
            parts = text.split(MARKER)
            pieces_embeds = [embeddings(ids(parts[0]))]
            for packet, part in zip(self.live_packets, parts[1:]):
                pieces_embeds.append(packet.squeeze(0).to(embeddings.weight.dtype))
                pieces_embeds.append(embeddings(ids(part)))
            embeds = torch.cat(pieces_embeds).unsqueeze(0)
        else:
            embeds = embeddings(ids(text)).unsqueeze(0)
        position = embeds.shape[1]
        out = self.b_model(
            inputs_embeds=embeds,
            position_ids=text_positions(1, position, self.device),
            use_cache=True,
        )
        past = out.past_key_values
        token = out.logits[:, -1].argmax(dim=-1)
        pieces: list[int] = []
        for _ in range(220):
            if token.item() == self.b_end:
                break
            pieces.append(token.item())
            out = self.b_model(
                input_ids=token.view(1, 1),
                position_ids=text_positions(1, 1, self.device, offset=position),
                past_key_values=past,
                use_cache=True,
            )
            position += 1
            past = out.past_key_values
            token = out.logits[:, -1].argmax(dim=-1)
        return self.b_tokenizer.decode(pieces).strip()

    def b_reply(self, text: str) -> str:
        self.b_history.append({"role": "user", "content": text})
        reply = self._b_generate()
        self.b_history.append({"role": "assistant", "content": reply})
        return reply


class ChatRequest(BaseModel):
    text: str = Field(max_length=4000)


@app.post("/api/chat_a")
def chat_a(request: ChatRequest) -> dict:
    with lock:
        instance = lab["lab"]
        reply = instance.a_reply(request.text)
        packets = instance.transmit(reply)
        return {"reply": reply, "packets": packets}


@app.post("/api/chat_b")
def chat_b(request: ChatRequest) -> dict:
    with lock:
        return {"reply": lab["lab"].b_reply(request.text)}


@app.post("/api/reset")
def reset() -> dict:
    with lock:
        lab["lab"].reset()
        return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Psychic Port</title>
<style>
:root{color-scheme:dark;--bg:#0b0f14;--panel:#121923;--line:#263244;--ink:#ecf2f8;--muted:#8fa2b7;--good:#73e6ba;--bad:#ff7d8b;--blue:#7cb7ff;--wire:#c9a5ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 0,#152133 0,var(--bg) 40%);color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}
main{max-width:1560px;margin:auto;padding:18px}
h1{font:700 22px ui-monospace,Menlo,monospace;margin:0 0 4px}.sub{color:var(--muted);margin-bottom:14px;font-size:13px}
.panes{display:grid;grid-template-columns:1fr 1.1fr 1fr;gap:14px}
.pane{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:12px;padding:14px;display:flex;flex-direction:column;min-height:600px;max-height:80vh}
.pane h2{font:700 13px ui-monospace,Menlo,monospace;margin:0 0 10px;letter-spacing:.08em}
.pane.a h2{color:var(--blue)}.pane.b h2{color:var(--good)}.pane.wire h2{color:var(--wire)}
.log{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding-right:4px}
.msg{border-radius:9px;padding:8px 11px;max-width:92%;white-space:pre-wrap;word-break:break-word}
.you{background:#1b2ab4aa;align-self:flex-end}.them{background:#0a1018;border:1px solid var(--line);align-self:flex-start}
.heard{background:#0d1a14;border:1px dashed #2c5e4a;align-self:flex-start;color:#a9e8cd;font-size:13px}
.wiremsg{background:#191129;border:1px solid #3c2a5e;border-radius:10px;padding:10px 12px;font:12px ui-monospace,Menlo,monospace}
.wiremsg .code{color:var(--wire);font-size:12.5px}
.gauges{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.chip{border-radius:99px;padding:2px 9px;font:11px ui-monospace,monospace;border:1px solid var(--line);color:var(--muted)}
.chip.ok{color:var(--good);border-color:#2c5e4a}.chip.bad{color:var(--bad);border-color:#5e2c39}.chip.hot{color:var(--wire);border-color:#3c2a5e}
.slots{display:grid;grid-template-columns:repeat(16,1fr);gap:3px;margin:8px 0}
.slot{height:20px;border-radius:4px;background:linear-gradient(135deg,#6d4fd1,#2c1a5e);animation:hum 2.2s ease-in-out infinite}
@keyframes hum{50%{filter:brightness(1.5)}}
.row{display:flex;gap:8px;margin-top:10px}
input{flex:1;background:#0a1018;border:1px solid var(--line);border-radius:8px;color:var(--ink);padding:9px;font:inherit}
button{border:0;border-radius:8px;padding:9px 13px;background:var(--blue);color:#07111e;font-weight:700;cursor:pointer;white-space:nowrap}
button.b{background:var(--good)}button.ghost{background:#243244;color:var(--ink)}
.empty{color:var(--muted);font-size:12.5px;text-align:center;margin:auto}
</style>
</head>
<body><main>
<h1>PSYCHIC PORT <span style="color:var(--muted)">— live</span></h1>
<div class="sub">Two real chats with two frozen models. Everything A says crosses to B as 16 vectors — no tokens between them. The middle pane is an independent wiretap translating the vectors directly; the dashed green bubbles are B's own reading of each transmission.</div>
<div class="panes">

<section class="pane a"><h2>◂ A — Qwen3-0.6B</h2>
  <div id="alog" class="log"><div class="empty">Say anything. A's replies cross the wire automatically.</div></div>
  <div class="row"><input id="ain" placeholder="talk to A…" onkeydown="if(event.key==='Enter')chatA()"><button onclick="chatA()">Send</button></div>
</section>

<section class="pane wire"><h2>⟿ THE WIRE — tapped</h2>
  <div id="wlog" class="log"><div class="empty">Quiet so far.</div></div>
  <div class="row"><button class="ghost" style="width:100%" onclick="resetAll()">reset everything</button></div>
</section>

<section class="pane b"><h2>B — Qwen3-VL-2B ▸</h2>
  <div id="blog" class="log"><div class="empty">B hears each transmission and reads it aloud here.</div></div>
  <div class="row"><input id="bin" placeholder="talk to B…" onkeydown="if(event.key==='Enter')chatB()"><button class="b" onclick="chatB()">Send</button></div>
</section>

</div></main>
<script>
const el=id=>document.getElementById(id);
fetch('/api/reset',{method:'POST'});
function bubble(log,cls,text){const d=document.createElement('div');d.className='msg '+cls;d.textContent=text;const box=el(log);if(box.querySelector('.empty'))box.innerHTML='';box.appendChild(d);box.scrollTop=box.scrollHeight}
async function post(path,body){const r=await fetch(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body||{})});return r.json()}
async function chatA(){
  const t=el('ain').value.trim();if(!t)return;el('ain').value='';bubble('alog','you',t);
  const d=await post('/api/chat_a',{text:t});
  bubble('alog','them',d.reply);
  const box=el('wlog');if(box.querySelector('.empty'))box.innerHTML='';
  for(const w of d.packets){
    const m=document.createElement('div');m.className='wiremsg';
    m.innerHTML='<div>A ⟿ B · packet '+w.packet+'</div>'
      +'<div class="slots">'+Array.from({length:16},(_,i)=>'<div class="slot" style="animation-delay:'+(i*0.13)+'s"></div>').join('')+'</div>'
      +'<div>tap reads: <span class="code">'+w.tap_read.replace(/</g,'&lt;')+'</span></div>'
      +'<div class="gauges">'
      +'<span class="chip '+(w.verdict==='LEGIT'?'ok':'bad')+'">'+w.verdict+' · conf '+w.confidence+' · z '+w.z+'</span>'
      +'<span class="chip hot">'+w.b_tokens_equivalent+' tokens → '+w.vectors+' vectors ('+w.density_vs_text+'×)</span>'
      +'</div>';
    box.appendChild(m);box.scrollTop=box.scrollHeight;
    bubble('blog','heard','⟿ B hears ('+w.packet+'): '+w.b_heard);
  }}
async function chatB(){const t=el('bin').value.trim();if(!t)return;el('bin').value='';bubble('blog','you',t);const d=await post('/api/chat_b',{text:t});bubble('blog','them',d.error||d.reply)}
async function resetAll(){await post('/api/reset');['alog','wlog','blog'].forEach(id=>el(id).innerHTML='<div class="empty">reset.</div>')}
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    lab["lab"] = Lab(args.device)
    print("psychic port v2 ready", flush=True)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
