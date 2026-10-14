"""Audio channel: faster-whisper (MIT) word-timestamped transcription ->
local-LLM PII spans -> per-word time ranges ->
  redact    : 1 kHz bleep (or silence) over the PII words
  transform : speak a same-type surrogate over the slot with a local TTS
              (Piper, optional: piper-tts>=1.3 is GPL-3.0 -> keep it an
              optional plug-in; falls back to bleep when unavailable).
Speaker VOICE anonymisation (kNN-VC / OpenVoice, MIT) is a roadmap hook.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

import hashlib

from .pii import redact_preview, sha, surrogate, value_key


@dataclass
class AudioEvent:
    t_start: float
    t_end: float
    label: str
    text: str
    surrogate: str
    action: str
    source: str
    surrogate_source: str = "local"
    surrogate_note: str | None = None   # guardian-replace: why the local fallback was used


def extract_audio(ffmpeg: str, video: str, out_wav: str, sr: int | None = None, mono=False) -> bool:
    cmd = [ffmpeg, "-v", "error", "-y", "-i", video, "-vn", "-acodec", "pcm_s16le"]
    if sr:
        cmd += ["-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    r = subprocess.run(cmd + [out_wav], capture_output=True)
    return r.returncode == 0 and Path(out_wav).exists() and Path(out_wav).stat().st_size > 1000


class AudioPII:
    def __init__(self, pii, whisper_model: str = "small.en", device: str = "cuda",
                 tts_voice: str | None = None, pad_s: float = 0.06, beep_hz: float = 1000.0):
        from faster_whisper import WhisperModel
        dev = "cuda" if device != "cpu" else "cpu"
        try:
            self.asr = WhisperModel(whisper_model, device=dev,
                                    compute_type="float16" if dev == "cuda" else "int8")
        except RuntimeError as e:  # shared GPU full -> CPU fallback
            print(f"[audio] whisper on {dev} failed ({e}); using CPU int8")
            dev = "cpu"
            self.asr = WhisperModel(whisper_model, device="cpu", compute_type="int8")
        self.name = f"faster-whisper {whisper_model} (MIT) on {dev}"
        self.pii, self.pad_s, self.beep_hz = pii, pad_s, beep_hz
        self.tts = None
        if tts_voice:
            try:
                from piper import PiperVoice
                self.tts = PiperVoice.load(tts_voice)
            except Exception as e:
                print(f"[audio] TTS voice unavailable ({e}); transform falls back to bleep")
        self.transcript = ""

    def transcribe(self, wav16k: str):
        segs, _ = self.asr.transcribe(wav16k, word_timestamps=True, vad_filter=False)
        words = []
        for s in segs:
            for w in (s.words or []):
                words.append((w.word, float(w.start), float(w.end)))
        return words

    def find_events(self, words, mode: str) -> list[AudioEvent]:
        text, offs = "", []
        for w, s, e in words:
            offs.append((len(text), len(text) + len(w), s, e))
            text += w
        self.transcript = text
        events = []
        for sp in self.pii.detect(text):
            hit = [(s, e) for a, b, s, e in offs if a < sp.end and b > sp.start]
            if not hit:
                continue
            action = "bleep" if mode == "redact" or self.tts is None else "surrogate_tts"
            events.append(AudioEvent(max(0.0, hit[0][0] - self.pad_s), hit[-1][1] + self.pad_s,
                                     sp.label, sp.text, sp.surrogate or surrogate(sp.label, sp.text),
                                     action, sp.source, sp.surrogate_source or "local",
                                     sp.surrogate_note))
        return events

    def _tts(self, text: str, sr: int) -> np.ndarray:
        chunks = list(self.tts.synthesize(text))
        a = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        src_sr = chunks[0].sample_rate
        n = int(len(a) * sr / src_sr)
        return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)

    def apply(self, wav_in: str, wav_out: str, events: list[AudioEvent]) -> None:
        a, sr = sf.read(wav_in, dtype="float32", always_2d=True)
        for ev in events:
            i0, i1 = int(ev.t_start * sr), min(len(a), int(ev.t_end * sr))
            n = i1 - i0
            if n <= 0:
                continue
            rms = float(np.sqrt(np.mean(a[i0:i1] ** 2)) + 1e-4)
            if ev.action == "surrogate_tts":
                s = self._tts(ev.surrogate, sr)
                # time-fit into the slot (resample; bounded to avoid chipmunking)
                ratio = np.clip(len(s) / n, 0.6, 1.6)
                m = int(len(s) / ratio)
                s = np.interp(np.linspace(0, len(s) - 1, m), np.arange(len(s)), s)
                s = np.pad(s, (0, max(0, n - m)))[:n]
                s *= rms / (np.sqrt(np.mean(s ** 2)) + 1e-6)
                rep = s
            else:
                t = np.arange(n) / sr
                rep = 0.25 * np.sin(2 * np.pi * self.beep_hz * t)
            fade = min(int(0.01 * sr), n // 4)
            env = np.ones(n, np.float32)
            if fade > 0:
                env[:fade] = np.linspace(0, 1, fade)
                env[-fade:] = np.linspace(1, 0, fade)
            a[i0:i1] = (rep * env)[:, None]
        sf.write(wav_out, np.clip(a, -1, 1), sr, subtype="PCM_16")

    @staticmethod
    def to_audit(ev: AudioEvent, log_text: bool = False) -> dict:
        return {"channel": "audio", "type": f"speech_{ev.label.lower()}", "pii_label": ev.label,
                "t_start": round(ev.t_start, 3), "t_end": round(ev.t_end, 3),
                "text_sha16": sha(ev.text), "text_preview": ev.text if log_text else redact_preview(ev.text),
                "value_key16": hashlib.sha256(value_key(ev.text).encode()).hexdigest()[:16],
                "action": ev.action, "surrogate": ev.surrogate if ev.action != "bleep" else None,
                "surrogate_planned": ev.surrogate, "surrogate_source": ev.surrogate_source,
                "surrogate_note": ev.surrogate_note,
                "detector": ev.source}
