"""Transcripción automática de las entrevistas con Whisper large-v3-turbo vía `transformers`
(usa el PyTorch ya instalado; no requiere las librerías CUDA 12 de faster-whisper).

Entrada: audio editado (solo el participante) features/audio_editado_denoised/*.wav (79).
         El 66 no tiene audio editado: se transcribe su audio del video
         (features/066_desde_video_PARA_EDITAR.wav), que INCLUYE al entrevistador; queda
         marcado como 'crudo' en el json.
Salida:  features/transcripciones/NNN.json (fragmentos con tiempos) y NNN.txt (texto)
         -> gitignored (datos clínicos). Reanudable por participante; sale al pasar --minutos.

Es lo mismo que podría hacer la plataforma con un paciente nuevo (reproducible), a diferencia
de las transcripciones corregidas a mano del servidor.

Uso:  python experimento_embeddings/transcribir.py
"""
import argparse
import json
import time
from pathlib import Path

import librosa
import torch
from transformers import pipeline

REPO = Path(__file__).resolve().parent.parent
AUDIO = REPO / "features" / "audio_editado_denoised"
AUDIO_66 = REPO / "features" / "066_desde_video_PARA_EDITAR.wav"
OUT = REPO / "features" / "transcripciones"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--minutos", type=float, default=6.5)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    tareas = [(int(w.name[:3]), w, "editado") for w in sorted(AUDIO.glob("*_recortado_denoised.wav"))]
    if AUDIO_66.exists():
        tareas.append((66, AUDIO_66, "crudo"))
    pendientes = [t for t in tareas if not (OUT / f"{t[0]:03d}.json").exists()]
    print(f"{len(tareas)} audios, {len(pendientes)} pendientes", flush=True)
    if not pendientes:
        return
    asr = pipeline("automatic-speech-recognition", model=args.modelo, torch_dtype=torch.float16,
                   device="cuda:0", chunk_length_s=30, batch_size=args.batch)
    for k, (pid, wav, tipo) in enumerate(pendientes, 1):
        if (time.time() - t0) / 60 > args.minutos:
            print(f"  tiempo agotado; quedan {len(pendientes) - k + 1}", flush=True)
            return
        y, _ = librosa.load(str(wav), sr=16000, mono=True)
        r = asr({"raw": y, "sampling_rate": 16000}, return_timestamps=True,
                generate_kwargs={"language": "spanish", "task": "transcribe"})
        frag = [{"ini": c["timestamp"][0], "fin": c["timestamp"][1], "texto": c["text"].strip()}
                for c in r.get("chunks", [])]
        texto = r["text"].strip()
        (OUT / f"{pid:03d}.json").write_text(json.dumps(
            {"pid": pid, "audio": tipo, "modelo": args.modelo, "duracion_s": round(len(y) / 16000, 1),
             "fragmentos": frag}, ensure_ascii=False, indent=1), encoding="utf-8")
        (OUT / f"{pid:03d}.txt").write_text(texto, encoding="utf-8")
        print(f"  {pid:03d} ({tipo}): {len(texto.split())} palabras  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
