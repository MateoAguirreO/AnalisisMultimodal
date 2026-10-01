"""Embeddings wav2vec2-large-robust por participante, con pooling DETERMINISTA.

Entrada: features/audio_crudo_demucs/NNN.wav (salida de preparar_audio_crudo.py).
Por participante:
  1. trim de silencio en los extremos (top_db=20, como la plataforma) y segmentos de
     5 s / 50% de solape (misma segmentación que eGeMAPS).
  2. cada segmento -> wav2vec2 -> hidden states de las 25 salidas (CNN + 24 capas);
     mean pooling sobre frames en cada capa. Sin pesos aleatorios: a diferencia de
     AttentionPooling en Psiquiatria (sin entrenar, sin semilla, sin guardar), esto es
     reproducible en la plataforma para un paciente nuevo.
  3. se guardan media y desviación sobre segmentos por capa, más el vector por
     segmento del promedio de capas 1..24 (representación principal, fijada a priori).

Salida: features/emb_w2v_large_robust/NNN.npz  (gitignored)
Reanudable. Uso:
  python experimento_embeddings/extraer_embeddings.py                       # audio crudo + Demucs
  python experimento_embeddings/extraer_embeddings.py --in-dir features/audio_editado_denoised \
         --out-dir features/emb_w2v_large_robust_editado                     # audio editado a mano
El código de participante se toma de los 3 primeros dígitos del nombre del wav
(sirve tanto para NNN.wav como para NNN_recortado_denoised.wav).
"""
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

REPO = Path(__file__).resolve().parent.parent
IN_DIR = REPO / "features" / "audio_crudo_demucs"
OUT_DIR = REPO / "features" / "emb_w2v_large_robust"
MODEL_ID = "facebook/wav2vec2-large-robust"
SR = 16000
SEG_S, OVERLAP, MIN_SEG_S = 5.0, 0.5, 1.0
VAD_TOP_DB = 20
BATCH = 16


def segmentar(y: np.ndarray) -> list[np.ndarray]:
    """Ventanas de 5 s con salto de 2.5 s; la última, si es parcial, se conserva sin
    padding (el padding con ceros sesgaría la media de frames) siempre que dure >= 1 s."""
    n, step = int(SEG_S * SR), int(SEG_S * SR * (1 - OVERLAP))
    if len(y) <= n:
        return [y]
    segs = []
    for start in range(0, len(y), step):
        seg = y[start:start + n]
        if len(seg) < MIN_SEG_S * SR:
            break
        segs.append(seg)
        if start + n >= len(y):
            break
    return segs


@torch.no_grad()
def embeber(segs, fe, model, device) -> np.ndarray:
    """-> array (n_seg, 25, 1024) con la media sobre frames de cada capa."""
    out = []
    completos = [s for s in segs if len(s) == int(SEG_S * SR)]
    parciales = [s for s in segs if len(s) != int(SEG_S * SR)]
    for grupo, bs in ((completos, BATCH), (parciales, 1)):
        for i in range(0, len(grupo), bs):
            lote = grupo[i:i + bs]
            x = fe(lote, sampling_rate=SR, return_tensors="pt").input_values.to(device)
            hs = model(x.half() if device.type == "cuda" else x,
                       output_hidden_states=True).hidden_states
            out.append(torch.stack([h.float().mean(dim=1) for h in hs], dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=IN_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    in_dir = args.in_dir if args.in_dir.is_absolute() else REPO / args.in_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID).to(device).eval()
    if device.type == "cuda":
        model = model.half()

    def destino(w: Path) -> Path:
        return out_dir / f"{int(w.name[:3]):03d}.npz"

    wavs = sorted(in_dir.glob("*.wav"))
    pendientes = [w for w in wavs if not destino(w).exists()]
    print(f"{len(wavs)} wav en {in_dir}, {len(pendientes)} pendientes, device={device}", flush=True)
    for k, w in enumerate(pendientes, 1):
        y, _ = librosa.load(str(w), sr=SR, mono=True)
        y_trim, _ = librosa.effects.trim(y, top_db=VAD_TOP_DB)
        y = y_trim if len(y_trim) > SR * 0.1 else y
        E = embeber(segmentar(y), fe, model, device)             # (n_seg, 25, 1024)
        np.savez_compressed(
            destino(w),
            mean_layers=E.mean(axis=0).astype(np.float32),        # (25, 1024)
            std_layers=E.std(axis=0).astype(np.float32),          # (25, 1024)
            seg_layeravg=E[:, 1:, :].mean(axis=1).astype(np.float16),  # (n_seg, 1024), capas 1..24
            n_seg=E.shape[0], dur_s=len(y) / SR,
        )
        if k % 10 == 0 or k == len(pendientes):
            print(f"  {k}/{len(pendientes)}  (último {w.stem}: {E.shape[0]} segmentos)", flush=True)
    print(f"OK -> {out_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
