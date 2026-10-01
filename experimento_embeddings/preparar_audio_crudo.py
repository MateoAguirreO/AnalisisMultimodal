"""wav CRUDO de cada participante -> Demucs (vocals, GPU) -> mono 16 kHz.

Es el mismo preprocesamiento que aplica PlataformaMultimodal/backend/extraction.py
(denoise_audio) a una muestra nueva. Lo que se entrene sobre este audio ve exactamente
lo mismo que vería la plataforma en inferencia.

OJO: NO es el audio con el que se calcularon features_*_egemaps.csv. Esos salen de
NNN_editado.wav (edición manual en el servidor, ~67% de la duración cruda en mediana),
ver auditoria_psiquiatria/AUDITORIA.md.

Salida: features/audio_crudo_demucs/NNN.wav (gitignored: dato clínico).
Reanudable: salta participantes ya procesados.

Uso:  python experimento_embeddings/preparar_audio_crudo.py
"""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MUESTRAS = REPO.parent / "analisis-espectrogramas" / ".MUESTRAS_SAMANÁ" / "Muestras"
OUT_DIR = REPO / "features" / "audio_crudo_demucs"
TMP_DIR = REPO / "features" / "_demucs_tmp"
SR = 16000
BATCH = 8  # archivos por llamada a demucs (el modelo se carga una vez por lote)


def mapa_wavs() -> dict[int, Path]:
    """codigo -> wav crudo. Cruza por CODIGO (carpeta NNN_cedula), nunca por cédula."""
    mapa = {}
    for carpeta in sorted(MUESTRAS.iterdir()):
        if not (carpeta.is_dir() and carpeta.name[:3].isdigit()):
            continue
        cod = int(carpeta.name[:3])
        wavs = [Path(f) for f in glob.glob(str(carpeta / "*.wav"))
                if Path(f).name.startswith(carpeta.name[:3])]
        if wavs:
            mapa[cod] = wavs[0]
    # 014.wav quedó guardado dentro de la carpeta del 016 (duración 204.9 s,
    # coincide con 014.mp4 = 211.3 s; la carpeta 014 no tiene wav).
    if 14 not in mapa:
        cand = glob.glob(str(MUESTRAS / "016_*" / "014.wav"))
        if cand:
            mapa[14] = Path(cand[0])
    return mapa


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mapa = mapa_wavs()
    pendientes = {c: w for c, w in mapa.items() if not (OUT_DIR / f"{c:03d}.wav").exists()}
    print(f"{len(mapa)} participantes con wav crudo; {len(pendientes)} pendientes", flush=True)

    items = sorted(pendientes.items())
    for i in range(0, len(items), BATCH):
        lote = items[i:i + BATCH]
        subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-d", "cuda",
                        "-o", str(TMP_DIR)] + [str(w) for _, w in lote],
                       check=True, capture_output=True, text=True)
        for cod, wav in lote:
            vocals = TMP_DIR / "htdemucs" / wav.stem / "vocals.wav"
            subprocess.run(["ffmpeg", "-y", "-i", str(vocals), "-ac", "1", "-ar", str(SR),
                            str(OUT_DIR / f"{cod:03d}.wav")],
                           check=True, capture_output=True, text=True)
            shutil.rmtree(vocals.parent)
        print(f"  {min(i + BATCH, len(items))}/{len(items)} listos", flush=True)

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(f"OK -> {OUT_DIR} ({len(os.listdir(OUT_DIR))} wav)", flush=True)


if __name__ == "__main__":
    main()
