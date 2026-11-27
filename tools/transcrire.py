#!/usr/bin/env python3
"""Télécharge les vidéos listées dans videos.txt, les transcrit (voix) et lit le texte affiché à l'écran (OCR).
Écrit transcriptions/<id>.txt et <id>.json, et tient à jour transcriptions/index.json."""
import json, os, re, subprocess, sys, glob, tempfile, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "transcriptions")
os.makedirs(OUT, exist_ok=True)
IDX = os.path.join(OUT, "index.json")
index = json.load(open(IDX)) if os.path.exists(IDX) else {}

def vid_id(url):
    m = re.findall(r"\d{6,}", url)
    return m[-1] if m else re.sub(r"\W+", "_", url)[-40:]

def fmt(t):
    t = int(t); return "%d:%02d:%02d" % (t // 3600, t % 3600 // 60, t % 60)

todo = []
for line in open(os.path.join(ROOT, "videos.txt"), encoding="utf-8"):
    line = line.strip()
    if not line or line.startswith("#"): continue
    url, _, label = line.partition(" ")
    if "fichier" in index.get(vid_id(url), {}) and not os.environ.get("FORCE"): continue
    todo.append((url, label.strip()))
if not todo:
    print("Rien de nouveau."); sys.exit(0)

from faster_whisper import WhisperModel
model = WhisperModel(os.environ.get("WHISPER_MODEL", "small"), device="cpu", compute_type="int8")

for url, label in todo:
    vid = vid_id(url); tmp = tempfile.mkdtemp()
    print("==>", url, vid, flush=True)
    try:
        r = subprocess.run(["yt-dlp", "-f", "b[height<=480]/b", "--no-playlist", "--no-simulate", "-o", os.path.join(tmp, "v.%(ext)s"),
                            "--print", "%(title)s|%(duration)s", url], capture_output=True, text=True)
        print(r.stdout, r.stderr[-1500:], flush=True)
        files = glob.glob(os.path.join(tmp, "v.*"))
        if r.returncode or not files:
            index[vid] = {"url": url, "label": label, "erreur": (r.stderr or "téléchargement impossible")[-300:]}; continue
        video = files[0]; title = (r.stdout.strip().splitlines() or [""])[0].split("|")[0]
        wav = os.path.join(tmp, "a.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", wav], check=True)
        segs, _ = model.transcribe(wav, language="fr", vad_filter=True, beam_size=1)
        speech = [{"t": round(s.start), "text": s.text.strip()} for s in segs]
        print("voix:", len(speech), "segments", flush=True)
        fr = os.path.join(tmp, "f"); os.makedirs(fr)
        step = int(os.environ.get("OCR_STEP", "4"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vf", "fps=1/%d,scale=1280:-1" % step, os.path.join(fr, "%05d.png")], check=True)
        ocr, last = [], ""
        for i, p in enumerate(sorted(glob.glob(os.path.join(fr, "*.png")))):
            txt = subprocess.run(["tesseract", p, "-", "-l", "fra", "--psm", "6"], capture_output=True, text=True).stdout
            lines = [l.strip() for l in txt.splitlines() if len(re.findall(r"[A-Za-zÀ-ÿ]", l)) >= 8]
            block = " / ".join(lines)
            if block and block != last: ocr.append({"t": i * step, "text": block}); last = block
        print("écran:", len(ocr), "blocs", flush=True)
        data = {"url": url, "label": label, "titre": title, "voix": speech, "ecran": ocr}
        json.dump(data, open(os.path.join(OUT, vid + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        with open(os.path.join(OUT, vid + ".txt"), "w", encoding="utf-8") as f:
            f.write("%s\n%s\n\n=== TEXTE À L'ÉCRAN ===\n" % (title or label, url))
            for o in ocr: f.write("[%s] %s\n" % (fmt(o["t"]), o["text"]))
            f.write("\n=== VOIX ===\n")
            for s in speech: f.write("[%s] %s\n" % (fmt(s["t"]), s["text"]))
        index[vid] = {"url": url, "label": label, "titre": title, "fichier": vid + ".txt"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        json.dump(index, open(IDX, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
