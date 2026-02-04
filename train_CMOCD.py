import warnings
import torch 
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11-RGBT/yolo11-RGBT6C-CoarseT_STN_DDF.yaml', task='detect')  # 只是将yaml里面的 ch设置成 6 ,红外部分改为 SilenceChannel, [ 3,6 ] 即可
    # model.load(r'best.pt') # loading pretrain weights
    #CMOCDwoCFFA,CMOCDwoVTFF
    #YOLO11 
    #VTFFA2 CFFA_STN  ACoarseT CoarseT 
    #ACoarseT_STN_DDF ACoarseT_PSTN_DDF CoarseT_PSTN_DDF PSTN_DDT ACoarseT_PSTN ACoarseT_DDF
    #CFFA_PerspectiveSTN CoarseT_PSTN CoarseT_PSTN_DDF PSTN_DDF CoarseT_STN_DDF
    device='3'
    name='yolo11-RGBT6C-CoarseT_STN_DDF-0814-Adamcos'
    model.train(data=R'./data/RGBT2723.yaml',
                cache=False,
                imgsz=1280, #1280,640
                epochs=300,
                batch=12,#8 12 16
                close_mosaic=0,
                workers=2,
                device=device,
                optimizer='Adam',  # using SGD ,Adam
                # resume='', # last.pt path
                amp=True, # close amp
                # fraction=0.2,
                use_simotm="RGBRGB6C",
                channels=6,  #
                # project='runs/try',
                project='runs/UAV2723-CMOCD-n-RGBT6c',
                name=name,
                lr0=1e-3,#SGD=1E-2, Adam=1E-3
                cos_lr=True,
                # step_size=30,
                # gamma=0.1,
                )
    
    # 训练完测试
    # print("测试模型",model.model)
    results = model.val(data='data/RGBT2723.yaml', split='test', 
                        save_txt=True, save_conf=True, 
                        # save_dir='runs/try/test_results', 
                        name=name,
                        imgsz=1280, 
                        device=device)