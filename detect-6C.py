import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('ultralytics/cfg/models/v8-RGBT/yolov8-RGBT-CMOCDwoVTFF.yaml', task='detect')
    # model.load(r'./runs/UAV2723-CMOCD-n-RGBT6c/yolov8-RGBT-CMOCDwoVTFF-1280-300e-/weights/best.pt')
    model = YOLO(r"/root/cly/YOLOv11-RGBT/runs/UAV2723-ppyoloe-s-RGBT6c/rgbt-ppyoloe-RGBT-midfusion-300-/weights/best.pt") # select your model.pt path
    
    results = model.val(data='data/RGBT2723.yaml', split='test', 
                        save_txt=True, save_conf=True, 
                        use_simotm="RGBRGB6C",
                        # save_dir='runs/try/test_results', 
                        name='TEST-ppyoloe-midfusion-',
                        imgsz=1280, 
                        channels=6, 
                        device='0')
    
#     model.predict(
#                   # data='data/RGBT2723.yaml', split='test', 
#                   source=r'/root/autodl-tmp/UAVCDdataset/test/visible/images',
#                   imgsz=1280,
#                   project='runs/detect',
#                   name='yolov8-RGBT-CMOCDwoVTFF-1280-300e-test-',
#                   show=True,
#                   save_frames=True,
#                   use_simotm="RGBRGB6C",
#                   channels=6,
#                   save=True,
#                   # conf=0.2,
#                   visualize=True # visualize model features maps
#                 )
    
    