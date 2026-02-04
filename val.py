import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'runs\yolov8-CoarseTfuse1\best.pt')
    model.val(data=r'data/exp.yaml',
              split='val',
              imgsz=640,
              batch=16,
              use_simotm="RGBRGB6C",
              channels=6,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val/try',
              name='yolo8n-RGBT-CoarseTfuse1-',
              )