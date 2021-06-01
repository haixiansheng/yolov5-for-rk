export PYTHONPATH="$PWD"
python3.8 models/export.py --weights yolov5s.pt --img 640 --batch 1
