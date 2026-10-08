"""Isolated installation acceptance, including real CPU LightGBM execution."""
from __future__ import annotations

import ctypes
import json
import sys
from importlib.metadata import entry_points
from pathlib import Path


def verify_lightgbm() -> dict:
    # Import LightGBM before torch: OpenMP must not depend on an earlier import.
    import lightgbm as lgb
    import numpy as np

    features = np.arange(80, dtype=np.float64).reshape(40, 2)
    labels = (features[:, 0] > 40).astype(np.int32)
    model = lgb.train({"objective": "binary", "verbosity": -1, "num_threads": 1, "min_data_in_leaf": 2}, lgb.Dataset(features, label=labels), num_boost_round=2)
    predictions = model.predict(features, num_threads=1)
    assert predictions.shape == (40,) and np.isfinite(predictions).all(), "LightGBM prediction smoke failed"
    loaded = []
    if sys.platform == "darwin":
        process = ctypes.CDLL(None)
        process._dyld_image_count.restype = ctypes.c_uint32
        process._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        process._dyld_get_image_name.restype = ctypes.c_char_p
        loaded = [process._dyld_get_image_name(index).decode() for index in range(process._dyld_image_count())]
        loaded = [item for item in loaded if Path(item).name == "libomp.dylib"]
        assert loaded and all(Path(item).resolve().is_relative_to(Path(sys.prefix).resolve()) for item in loaded), f"OpenMP escaped the engine runtime: {loaded}"
    return {"lightgbm": lgb.__version__, "cpu_prediction_count": len(predictions), "openmp_libraries": loaded}


def main() -> None:
    report = verify_lightgbm()
    if "--lightgbm-only" not in sys.argv:
        import biomaker.run  # noqa: F401
        import torch
        import monai

        assert any(item.name == "biomaker" and item.value == "biomaker.run:run" for item in entry_points(group="console_scripts")), "BioMaker CLI entrypoint is missing"
        report.update(torch=torch.__version__, monai=monai.__version__)
    print(json.dumps(report, ensure_ascii=False))
    print("BioMaker engine is ready.")


if __name__ == "__main__":
    main()
