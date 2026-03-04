from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

@dataclass
class ESC50AudioConfig:
    sample_rate: int = 16000
    clip_duration_s: float = 5.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float | None = 8000.0


def resolve_esc50_paths(data_root: str | Path) -> tuple[Path, Path]:
    root = Path(data_root)
    csv_candidates = [
        root / "esc50.csv",
        root / "esc-50.csv",
        root / "ESC-50.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.is_file()), None)
    if csv_path is None:
        expected_csv = ", ".join(p.name for p in csv_candidates)
        raise FileNotFoundError(
            f"Could not find metadata CSV in {root}. "
            f"Expected one of: {expected_csv}."
        )

    audio_dir = root / "audio"
    if not audio_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find audio directory at {audio_dir}. "
            f"Expected dataset layout: {root}/audio and {root}/{csv_path.name}."
        )
    return csv_path, audio_dir


class ESC50Dataset(Dataset):
    """ESC-50 dataset loader producing normalized log-mel spectrograms."""

    def __init__(
        self,
        data_root: str | Path,
        folds: Sequence[int],
        audio_config: ESC50AudioConfig | None = None,
        random_crop: bool = False,
    ) -> None:
        if torchaudio is None:
            raise ImportError(
                "torchaudio is required for audio loading/transforms. "
                "Install dependencies with: pip install -r requirements.txt"
            )
        self.csv_path, self.audio_dir = resolve_esc50_paths(data_root)
        self.audio_config = audio_config or ESC50AudioConfig()
        self.random_crop = random_crop
        self.target_num_samples = int(
            self.audio_config.sample_rate * self.audio_config.clip_duration_s
        )
        self.metadata = pd.read_csv(self.csv_path)
        self.metadata = self.metadata[self.metadata["fold"].isin(folds)].reset_index(
            drop=True
        )
        self._resamplers: dict[int, object] = {}

        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.audio_config.sample_rate,
            n_fft=self.audio_config.n_fft,
            hop_length=self.audio_config.hop_length,
            n_mels=self.audio_config.n_mels,
            f_min=self.audio_config.f_min,
            f_max=self.audio_config.f_max,
            center=True,
            power=2.0,
        )
        self._db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self) -> int:
        return len(self.metadata)

    def _resample_if_needed(self, audio: torch.Tensor, source_sr: int) -> torch.Tensor:
        target_sr = self.audio_config.sample_rate
        if source_sr == target_sr:
            return audio
        if source_sr not in self._resamplers:
            self._resamplers[source_sr] = torchaudio.transforms.Resample(
                orig_freq=source_sr,
                new_freq=target_sr,
            )
        return self._resamplers[source_sr](audio)

    def _fix_length(self, audio: torch.Tensor) -> torch.Tensor:
        total = audio.shape[-1]
        target = self.target_num_samples
        if total > target:
            max_start = total - target
            if self.random_crop:
                start = int(torch.randint(0, max_start + 1, (1,)).item())
            else:
                start = max_start // 2
            audio = audio[..., start : start + target]
        elif total < target:
            audio = F.pad(audio, (0, target - total))
        return audio

    @staticmethod
    def _peak_normalize(audio: torch.Tensor) -> torch.Tensor:
        peak = audio.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        return audio / peak

    def _to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        mel = self._mel_transform(audio)
        mel = self._db_transform(mel)
        mel = mel.squeeze(0)
        mel = (mel - mel.mean()) / mel.std().clamp_min(1e-6)
        return mel

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.metadata.iloc[index]
        audio_path = self.audio_dir / row["filename"]
        audio, source_sr = torchaudio.load(str(audio_path))
        audio = audio.mean(dim=0, keepdim=True)
        audio = self._resample_if_needed(audio, source_sr)
        audio = self._fix_length(audio)
        audio = self._peak_normalize(audio)
        target = int(row["target"])

        spectrogram = self._to_spectrogram(audio)
        return spectrogram, target
