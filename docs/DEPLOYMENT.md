# Deployment
Only local Python inference exists:
```
pip install -r requirements.txt
python src/models/predict.py --image leaf.jpg --checkpoint <ckpt.pth> [--temperature T] [--threshold 0.7]
```
Place `calibration.json` (`{"temperature": <float>}`) next to the checkpoint; otherwise a warning is printed and the default T is used. Exit with a clear message if the checkpoint cannot be loaded. No Docker, CI, mobile build or API exists yet.
