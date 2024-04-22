import shutil
from pathlib import Path

import cv2
import numpy as np
from videoToFrames.breaker import VideoToFrames

from Config import Config


def _divide_video_to_frames(
    video_path: Path,
    out_path: Path = Config.temp_output_video_folder,
    load_to_mem: bool = True,
) -> np.ndarray:
    obj = VideoToFrames(path=str(video_path))
    obj.run_breaking(out_path)
    if load_to_mem:
        try:
            return np.array(
                tuple(
                    map(
                        cv2.imread,
                        map(
                            str,
                            sorted(
                                out_path.iterdir(),
                                key=lambda path: int(
                                    path.name.split(".")[0].strip("_")
                                ),
                            ),
                        ),
                    ),
                )
            )
        finally:
            shutil.rmtree(out_path)
    raise ValueError("Not implemented yet")


if __name__ == "__main__":
    _divide_video_to_frames(Config.datasets_path.joinpath("2024-03-21 16-50-53.mkv"))
