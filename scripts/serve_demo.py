"""本地网页演示：听障无障碍环境声音提示智能体。

只用 Python 标准库启动 HTTP 服务，不依赖任何 Web 框架。
浏览器打开后可选音频样本、调阈值，实时看到识别结果与提示文案。

用法:
    python scripts/serve_demo.py            # 默认 http://127.0.0.1:8000
    python scripts/serve_demo.py --port 8080
"""
import argparse
import csv
import io
import json
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audio_io import load_audio            # noqa: E402
from decision import decide                # noqa: E402
from event_mapping import build_lookup_from_csv  # noqa: E402
from yamnet_runner import YamnetRunner     # noqa: E402

SAMPLES_DIR = ROOT / "data" / "samples"
MODEL_PATH = ROOT / "models" / "yamnet.onnx"
CLASS_MAP = ROOT / "models" / "yamnet_class_map.csv"
COZE_OUTPUTS = ROOT / "data" / "out" / "coze_outputs.json"

_lock = threading.Lock()
_state: dict = {}


def load_state() -> dict:
    """加载模型与类别表，只执行一次。"""
    if _state:
        return _state
    with _lock:
        if _state:
            return _state
        if not MODEL_PATH.exists():
            raise SystemExit(
                f"缺少模型 {MODEL_PATH}\n请先运行: python scripts/download_model.py"
            )
        from download_model import parse_class_map

        _state["class_names"] = parse_class_map(CLASS_MAP)
        _state["lookup"] = build_lookup_from_csv(CLASS_MAP)
        _state["runner"] = YamnetRunner(MODEL_PATH)
        _state["cache"] = {}
    return _state


def load_manifest() -> list[dict]:
    path = SAMPLES_DIR / "manifest.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_coze() -> dict:
    if not COZE_OUTPUTS.exists():
        return {}
    return json.loads(COZE_OUTPUTS.read_text(encoding="utf-8"))


def detect(filename: str, threshold: float) -> dict:
    """识别单个样本，结果按 (文件名, 阈值) 缓存。"""
    state = load_state()
    key = (filename, round(threshold, 3))
    if key in state["cache"]:
        return state["cache"][key]

    path = SAMPLES_DIR / f"{filename}.wav"
    waveform, sr = load_audio(path)
    frame_scores = state["runner"].score(waveform)
    scores = frame_scores.mean(axis=0)
    events = decide(scores, state["lookup"], threshold, state["class_names"])

    peak = scores.max() if scores.size else 0.0
    result = {
        "file": filename,
        "duration": round(len(waveform) / sr, 2),
        "n_frames": int(frame_scores.shape[0]),
        "max_level": min((e.level for e in events), default=0),
        "events": [e.to_dict() for e in events],
        "top_score": round(float(peak), 4),
        "waveform": waveform[:: max(1, len(waveform) // 600)].round(4).tolist(),
    }
    state["cache"][key] = result
    return result


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb


def mel_spectrogram_png(waveform, out_path: Path) -> None:
    """画波形与频谱图，存为 PNG。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.2))
    t = [i / 16000 for i in range(len(waveform))]
    axes[0].plot(t, waveform, linewidth=0.6, color="#2E5C8A")
    axes[0].set_ylabel("幅值")
    axes[0].set_title("波形", fontsize=10)
    axes[0].set_xlim(0, t[-1] if t else 1)

    f, tt, Sxx = spectrogram(waveform, fs=16000, nperseg=512, noverlap=384)
    axes[1].pcolormesh(tt, f, 10 * __import__("numpy").log10(Sxx + 1e-10),
                       shading="gouraud", cmap="magma")
    axes[1].set_ylabel("频率 (Hz)")
    axes[1].set_xlabel("时间 (秒)")
    axes[1].set_title("频谱", fontsize=10)
    axes[1].set_ylim(0, 8000)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>听障无障碍环境声音提示智能体</title>
<style>
 *{box-sizing:border-box}
 body{margin:0;font-family:"Microsoft YaHei",system-ui,sans-serif;background:#f5f6f8;color:#1c1e21}
 header{background:#1c3d5a;color:#fff;padding:14px 24px;font-size:17px;font-weight:600}
 header small{font-weight:400;opacity:.75;margin-left:10px;font-size:13px}
 .wrap{display:flex;height:calc(100vh - 52px)}
 aside{width:270px;background:#fff;border-right:1px solid #e2e5e9;overflow-y:auto;flex:none}
 aside h3{margin:0;padding:12px 16px;font-size:12px;color:#667;letter-spacing:.5px;
          border-bottom:1px solid #eef0f3;position:sticky;top:0;background:#fff}
 .item{padding:9px 16px;cursor:pointer;font-size:13px;border-bottom:1px solid #f4f5f7}
 .item:hover{background:#eef4fb}
 .item.on{background:#dce9f7;font-weight:600;color:#14406b}
 .item span{color:#889;font-size:11px;float:right}
 main{flex:1;overflow-y:auto;padding:18px 24px}
 .bar{display:flex;align-items:center;gap:14px;background:#fff;padding:12px 16px;
      border-radius:8px;margin-bottom:14px;font-size:13px;border:1px solid #e6e9ed}
 .bar input[type=range]{flex:1;max-width:320px}
 .bar b{color:#14406b;font-variant-numeric:tabular-nums}
 .card{background:#fff;border-radius:8px;padding:16px 18px;margin-bottom:14px;
       border:1px solid #e6e9ed}
 .card h2{margin:0 0 12px;font-size:14px;color:#14406b}
 img{width:100%;border-radius:6px;display:block}
 .ev{display:flex;align-items:center;gap:12px;padding:9px 0;border-bottom:1px solid #f4f5f7}
 .ev:last-child{border-bottom:none}
 .lv{width:46px;text-align:center;border-radius:4px;font-size:11px;padding:3px 0;color:#fff;flex:none}
 .l1{background:#c0392b}.l2{background:#d68910}
 .nm{width:110px;font-size:13px}
 .bar-bg{flex:1;background:#eef0f3;height:9px;border-radius:5px;overflow:hidden}
 .bar-fg{height:100%;border-radius:5px}
 .cf{width:52px;text-align:right;font-size:12px;color:#667;font-variant-numeric:tabular-nums}
 .prompt{background:#f0f7ff;border-left:3px solid #2E5C8A;padding:12px 14px;
         border-radius:0 6px 6px 0;font-size:14px;line-height:1.7}
 pre{background:#23272e;color:#d8dee9;padding:12px 14px;border-radius:6px;
     font-size:11.5px;line-height:1.55;overflow-x:auto;margin:0;max-height:280px}
 .empty{color:#99a;font-size:13px;padding:8px 0}
 .meta{font-size:12px;color:#778;margin-bottom:10px}
 .tag{display:inline-block;background:#eef4fb;color:#2E5C8A;border-radius:3px;
      padding:1px 7px;font-size:11px;margin-right:6px}
</style></head><body>
<header>听障无障碍环境声音提示智能体<small>YAMNet 零样本声学事件检测 + Coze 提示生成</small></header>
<div class="wrap">
 <aside><h3>音频样本（ESC-50 抽样）</h3><div id="list"></div></aside>
 <main>
  <div class="bar">检测阈值 <input type="range" id="th" min="0.05" max="0.5"
      step="0.05" value="0.10"><b id="thv">0.10</b>
      <span style="color:#889" id="hint">拖动可观察召回与误报的权衡</span></div>
  <div class="card"><h2>识别结果</h2><div id="res"><div class="empty">← 请从左侧选择一个音频样本</div></div></div>
  <div class="card"><h2>波形与频谱</h2><img id="plot" alt="" style="display:none"></div>
  <div class="card"><h2>发送给 Coze 的事件 JSON</h2><pre id="json">—</pre></div>
  <div class="card"><h2>Coze 生成的提示文案</h2><div id="prompt"><div class="empty">—</div></div></div>
 </main>
</div>
<script>
let cur=null, th=0.10, coze={};
const LV={1:['l1','危险'],2:['l2','注意']};
fetch('/api/samples').then(r=>r.json()).then(d=>{
  coze=d.coze;
  const byCat={};
  d.samples.forEach(s=>(byCat[s.category]=byCat[s.category]||[]).push(s));
  document.getElementById('list').innerHTML=Object.keys(byCat).sort().map(c=>
    `<h3>${c} (${byCat[c].length})</h3>`+byCat[c].map(s=>
      `<div class="item" data-f="${s.filename}">${s.filename}<span>${s.fold}</span></div>`
    ).join('')).join('');
  document.querySelectorAll('.item').forEach(el=>el.onclick=()=>pick(el.dataset.f,el));
});
document.getElementById('th').oninput=e=>{
  th=parseFloat(e.target.value);
  document.getElementById('thv').textContent=th.toFixed(2);
  if(cur) render();
};
function pick(f,el){
  cur=f;
  document.querySelectorAll('.item').forEach(x=>x.classList.remove('on'));
  el.classList.add('on');
  document.getElementById('plot').src='/plot?file='+encodeURIComponent(f)+'&_='+Date.now();
  document.getElementById('plot').style.display='block';
  render();
}
function render(){
  if(!cur) return;
  fetch(`/api/detect?file=${encodeURIComponent(cur)}&threshold=${th}`)
   .then(r=>r.json()).then(d=>{
    const box=document.getElementById('res');
    if(!d.events.length){
      box.innerHTML=`<div class="empty">无提示事件（阈值 ${th.toFixed(2)} 下未检出）</div>`;
    } else {
      box.innerHTML=`<div class="meta">时长 ${d.duration}s ｜ ${d.n_frames} 帧 ｜ 最高等级 `+
        (d.max_level?LV[d.max_level][1]:'—')+`</div>`+
        d.events.map(e=>{
          const [cls,name]=LV[e.level]||['l2','—'];
          const pct=(e.confidence*100).toFixed(1);
          return `<div class="ev"><div class="lv ${cls}">${name}</div>
            <div class="nm">${e.name_cn}</div>
            <div class="bar-bg"><div class="bar-fg" style="width:${pct}%;background:${
              e.level==1?'#c0392b':'#d68910'}"></div></div>
            <div class="cf">${e.confidence.toFixed(2)}</div></div>`;
        }).join('');
    }
    document.getElementById('json').textContent=JSON.stringify({
      session_id:cur, audio_file:cur+'.wav', duration_sec:d.duration,
      max_level:d.max_level, events:d.events}, null, 2);
    const c=coze[cur];
    document.getElementById('prompt').innerHTML = c
      ? `<div class="prompt">${c.output}</div>
         <div class="meta" style="margin-top:8px"><span class="tag">Coze 实测输出</span>
         该样本已实测验证，结果低于阈值时会显示既有文案</div>`
      : `<div class="empty">该样本未做 Coze 实测。可把上方 JSON 粘贴进 Coze 大模型节点获取提示文案。</div>`;
   });
}
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # 静默，避免终端被请求日志刷屏

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", code)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/":
            self._send(HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/api/samples":
            self._json({"samples": load_manifest(), "coze": load_coze()})
        elif parsed.path == "/api/detect":
            name = (q.get("file") or [""])[0]
            try:
                th = float((q.get("threshold") or ["0.1"])[0])
                self._json(detect(name, th))
            except Exception as exc:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        elif parsed.path == "/plot":
            name = (q.get("file") or [""])[0]
            path = SAMPLES_DIR / f"{name}.wav"
            if not path.exists():
                self._send(b"not found", "text/plain", 404)
                return
            out = ROOT / "data" / "out" / f"_demo_{name}.png"
            try:
                waveform, _ = load_audio(path)
                mel_spectrogram_png(waveform, out)
                self._send(out.read_bytes(), "image/png")
            except Exception as exc:
                self._send(str(exc).encode("utf-8"), "text/plain", 500)
        else:
            self._send(b"not found", "text/plain", 404)


def main() -> int:
    parser = argparse.ArgumentParser(description="本地网页演示")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    samples = load_manifest()
    if not samples:
        print("缺少样本清单，请先运行: python scripts/download_samples.py",
              file=sys.stderr)
        return 1

    print("正在加载模型…")
    load_state()
    print(f"样本 {len(samples)} 条，模型已就绪")
    print(f"\n演示地址： http://{args.host}:{args.port}\n以 Ctrl+C 结束\n")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
