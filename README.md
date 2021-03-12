# YOLO
This repo is based on [Jittor](https://github.com/Jittor/jittor). It supports yolov3 and yolov5.

## Test Results
**Pytorch(coco128)**
|Model|Class|Images|Targets|P|R|mAP@.5|mAP@.5:.95|
|----|----|----|----|----|----|----|----|
|yolov3-spp|all|128|929|0.549|0.868| 0.82|0.574|
|yolov3|all|128|929|0.527|0.831|0.791|0.56|
|yolov3-tiny|all|128|929|0.399|0.435|0.442|0.229|
|yolov5s|all|128|929|0.641|0.637|0.656|0.429|
|yolov5m|all|128|929|0.775|0.702|0.788|0.569|
|yolov5l|all|128|929|0.8|0.743|0.821|0.614|
|yolov5x|all|128|929|0.829|0.757|0.857|0.642|


**Jittor(coco128)**
|Model|Class|Images|Targets|P|R|mAP@.5|mAP@.5:.95|
|----|----|----|----|----|----|----|----|
|[yolov3-spp](https://cloud.tsinghua.edu.cn/d/69b55d71e7ff46978a65/files/?p=%2Fyolo%2Fyolov3-spp.pkl&dl=1)|all|128|929|0.551|0.869|0.821|0.574|
|[yolov3](https://cloud.tsinghua.edu.cn/d/69b55d71e7ff46978a65/files/?p=%2Fyolo%2Fyolov3.pkl&dl=1)|all|128|929| 0.53|0.829| 0.79|0.561|
|[yolov3-tiny](https://cloud.tsinghua.edu.cn/d/69b55d71e7ff46978a65/files/?p=%2Fyolo%2Fyolov3-tiny.pkl&dl=1)|all|128|929|0.454| 0.44| 0.45|0.234|
|[yolov5s](https://cloud.tsinghua.edu.cn/d/69b55d71e7ff46978a65/files/?p=%2Fyolo%2Fyolov5s.pkl&dl=1)|all|128|929|0.639|0.637|0.655|0.427|
|[yolov5m](https://cloud.tsinghua.edu.cn/d/69b55d71e7ff46978a65/files/?p=%2Fyolo%2Fyolov5m.pkl&dl=1)|all|128|929|0.769|0.707|0.787|0.569|
|[yolov5l](https://cloud.tsinghua.edu.cn/d/69b55d71e7ff46978a65/files/?p=%2Fyolo%2Fyolov5l.pkl&dl=1)|all|128|929|0.801|0.742|0.821|0.613|
|[yolov5x](https://cloud.tsinghua.edu.cn/d/69b55d71e7ff46978a65/files/?p=%2Fyolo%2Fyolov5x.pkl&dl=1)|all|128|929| 0.83|0.756|0.857|0.642|

**All Checkpoint links is [here](https://cloud.tsinghua.edu.cn/d/69b55d71e7ff46978a65/?p=%2Fyolo&mode=list)**


## Reference
1. https://github.com/Jittor/jittor
2. https://github.com/ultralytics/yolov3
3. https://github.com/ultralytics/yolov5