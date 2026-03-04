from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf


def build_sound(
    frequncy_amplitude_list: list[tuple[float, float]],
    duration: float,
    prefix: str | None = None,
) -> None:
    sample_rate = 44_100
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename_prefix = prefix or timestamp
    output_path = Path("audio") / f"sound_{filename_prefix}.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_count = int(sample_rate * duration)
    t = np.linspace(0.0, duration, sample_count, endpoint=False)
    segments: list[np.ndarray] = []

    for frequency, amplitude in frequncy_amplitude_list:
        note_waveform = amplitude * np.sin(2.0 * np.pi * frequency * t)
        segments.append(note_waveform)

    combined_waveform = np.zeros(sample_count, dtype=np.float64)
    for frequency, amplitude in frequncy_amplitude_list:
        combined_waveform += amplitude * np.sin(2.0 * np.pi * frequency * t)
    segments.append(combined_waveform)

    waveform = np.concatenate(segments)

    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform /= peak

    sf.write(output_path, waveform, sample_rate)
    print(f"Saved: {output_path}")


def build_sound_for_ratio(rational: tuple[int, int]):
    base = 256
    upper = base * rational[0] / rational[1]
    prefix = "ratio_" + str(rational[0]) + "_" + str(rational[1])
    build_sound([(base, 1), (upper, 1)], 2, prefix)


if __name__ == "__main__":
    for rational in [
        # 5
        (3, 2),
        # 7
        (4, 3),
        # 8
        (5, 3),
        # 9
        (5, 4),
        # 11
        (6, 5),
        (7, 4),
    ]:
        assert (
            1 < rational[0] / rational[1] < 2
        ), "The ratio should be between 1 and 2."

        build_sound_for_ratio(rational)

    build_sound(
        [(256, 1), (256 * 5 / 4, 1), (256 * 3 / 2, 1), (256 * 7 / 4, 1)],
        1,
        "dominant_7th_ish",
    )
    build_sound(
        [(256, 1), (256 * 5 / 4, 1), (256 * 3 / 2, 1), (256 * 16 / 9, 1)],
        1,
        "dominant_7th",
    )
