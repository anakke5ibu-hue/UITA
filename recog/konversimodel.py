from ultralytics import YOLO

model = YOLO('C:\\Users\\syawal\\Downloads\\UI TA\\face-recognition-web\\recog\\model.pt')
print(model.ckpt['train_args']['epochs'])