## YOLO V9 기반 객체탐지

### 준비

- weights 받아오려면 git lfs 필요함
- 의존성은 requirments.txt에 있는거만 쓰면 됌

### 학습

```
python yolov9/train.py \
--batch 16 --epochs 25 --img 640 --device 0 --min-items 0 --close-mosaic 15 \
--data data/roboflow/data.yaml \
--weights weights/gelan-c.pt \
--cfg yolov9/models/detect/gelan-c.yaml \
--hyp hyp.scratch-high.yaml
```
