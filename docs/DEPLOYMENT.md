# Deployment (current scope: local Python only)
```
pip install -r requirements-lock.txt
python src/models/predict.py --image leaf.jpg --checkpoint reports/mobilenetv3_baseline/best_model.pth
```
Requires beside the checkpoint a `calibration.json` produced by `calibrate.py` + `select_threshold.py` (bound to the checkpoint sha256). Missing/mismatched artifact => error (or `--allow_uncalibrated`, which never returns `accepted`). Output includes `status` (accepted | low_confidence | invalid_image), top-3, `calibrated`, advisory image quality, and the non-diagnosis disclaimer.
ONNX: `python src/models/export_onnx.py` (real checkpoints only; writes `.onnx.parity.json`).
Not available yet: TFLite/INT8, Android, backend, Docker, CI.
