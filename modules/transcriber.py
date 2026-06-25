"""
音声→テキスト変換モジュール（Whisper API）。

2つのモードに対応:
  1. ファイルモード: 録音済みファイルをWhisper APIに送信
  2. チャンクモード: sounddeviceで一定間隔録音→逐次送信（擬似リアルタイム）
"""

import io
import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SAMPLE_RATE = 16000
CHUNK_SECONDS = 8       # 何秒ごとにWhisperへ送るか
SILENCE_THRESHOLD = 0.01


def transcribe_file(filepath, language: str = "ja") -> str:
    """録音済みファイルをWhisper APIで文字起こし"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(filepath, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language=language,
            response_format="text",
        )
    return result.strip()


def transcribe_audio_bytes(audio_bytes: bytes, language: str = "ja") -> str:
    """バイト列（WAV形式）をWhisper APIで文字起こし"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.wav"
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language=language,
        response_format="text",
    )
    return result.strip()


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """numpy配列をWAVバイト列に変換"""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


class RealtimeTranscriber:
    """
    マイクから音声をチャンク録音し、Whisper APIで逐次文字起こしする。

    使い方:
        def on_text(text: str):
            print(f"文字起こし: {text}")

        t = RealtimeTranscriber(on_transcription=on_text)
        t.start()
        time.sleep(60)  # 録音時間
        t.stop()
    """

    def __init__(
        self,
        on_transcription: Callable[[str], None],
        chunk_seconds: int = CHUNK_SECONDS,
        language: str = "ja",
    ):
        self.on_transcription = on_transcription
        self.chunk_seconds = chunk_seconds
        self.language = language
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running = False
        self._record_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._record_thread.start()
        self._process_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._record_thread:
            self._record_thread.join(timeout=3)
        if self._process_thread:
            self._process_thread.join(timeout=3)

    def _record_loop(self) -> None:
        """一定秒数ごとに録音してキューに積む"""
        while self._running:
            frames = int(SAMPLE_RATE * self.chunk_seconds)
            audio = sd.rec(
                frames,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            if not self._running:
                break
            if np.abs(audio).max() > SILENCE_THRESHOLD:
                self._audio_queue.put(audio.flatten())

    def _process_loop(self) -> None:
        """キューから音声を取り出してWhisperへ送る"""
        while self._running:
            try:
                audio = self._audio_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                wav_bytes = numpy_to_wav_bytes(audio)
                text = transcribe_audio_bytes(wav_bytes, language=self.language)
                if text:
                    self.on_transcription(text)
            except Exception as e:
                print(f"[transcriber] error: {e}")


def parse_plaud_transcript(raw_text: str) -> List[dict]:
    """
    PLAUD NOTE等からエクスポートしたテキストを発言リストに変換する。

    対応フォーマット例:
        00:00:12 部長: 今期の売上どうなってる？
        00:00:28 営業: 顧客Aの案件、来月クロージング予定です

    Returns:
        [{"timestamp": "00:00:12", "speaker": "部長", "text": "今期の..."}, ...]
    """
    utterances = []
    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # タイムスタンプ付きフォーマット: "HH:MM:SS 発言者: テキスト"
        parts = line.split(" ", 1)
        if len(parts) == 2 and ":" in parts[1]:
            timestamp = parts[0]
            rest = parts[1]
            if ": " in rest:
                speaker, text = rest.split(": ", 1)
                utterances.append({
                    "timestamp": timestamp,
                    "speaker": speaker.strip(),
                    "text": text.strip(),
                })
                continue
        # タイムスタンプなしフォーマット: "発言者: テキスト"
        if ": " in line:
            speaker, text = line.split(": ", 1)
            utterances.append({
                "timestamp": "",
                "speaker": speaker.strip(),
                "text": text.strip(),
            })
    return utterances
