"""eGeMAPS por segmento sobre el audio CRUDO + Demucs, con la MISMA función que usa
la plataforma en inferencia (PlataformaMultimodal/backend/extraction.py::audio_to_segments).

Sirve para medir cuánto pierde la rama de voz desplegada: se entrenó con eGeMAPS de
audio editado a mano (features_*_egemaps.csv), pero la plataforma le entrega eGeMAPS
de audio crudo.

Salida: features/egemaps_audio_crudo.csv  (pid, seg_idx, 88 features)
Uso:  python experimento_embeddings/egemaps_audio_crudo.py
"""
import importlib.util
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
IN_DIR = REPO / "features" / "audio_crudo_demucs"
OUT = REPO / "features" / "egemaps_audio_crudo.csv"
EXTRACTION = REPO.parent / "PlataformaMultimodal" / "backend" / "extraction.py"


def _cargar_extraction():
    spec = importlib.util.spec_from_file_location("extraction_plataforma", EXTRACTION)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ext = _cargar_extraction()
    filas = []
    wavs = sorted(IN_DIR.glob("*.wav"))
    for k, w in enumerate(wavs, 1):
        for i, feats in enumerate(ext.audio_to_segments(w)):
            filas.append({"pid": int(w.stem), "seg_idx": i, **feats})
        if k % 10 == 0 or k == len(wavs):
            print(f"  {k}/{len(wavs)}", flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(OUT, index=False)
    print(f"OK -> {OUT}  ({df.pid.nunique()} participantes, {len(df)} segmentos)")


if __name__ == "__main__":
    main()
