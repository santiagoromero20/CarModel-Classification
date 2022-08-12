
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from torchvision.ops import box_area
from detectron2.data import MetadataCatalog
import cv2

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.DEVICE='cpu'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
DET_MODEL = DefaultPredictor(cfg)

def get_vehicle_coordinates(img):
   outputs   = DET_MODEL(img)
   instances = outputs["instances"]
   detected_class_indexes = instances.pred_classes
   prediction_boxes = instances.pred_boxes

   metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
   class_catalog = metadata.thing_classes

   imp_classes = []
   all_classes = []
   for idx, coordinates in enumerate(prediction_boxes):
      class_index = detected_class_indexes[idx]
      class_name  = class_catalog[class_index]
      all_classes.append((class_name, coordinates))

   sum = 0
   for i in range(len(all_classes)):
      if all_classes[i][0] == "car" or all_classes[i][0] == "truck":
         imp_classes.append(all_classes[i][1])
         sum = sum + 1

   if sum != 0:
      bbox = imp_classes[0]
      bbox = bbox.unsqueeze(0)
      max_area  = box_area(bbox)
      max_area = max_area[0].item()
   
      pos = 0
      for i in range(len(imp_classes)):
         bbox1 = imp_classes[i]
         bbox1 = bbox1.unsqueeze(0)
         area = box_area(bbox1)
         area = area[0].item()
         if area >= max_area:
            max_area = area
            pos = i

         biggest_area = imp_classes[pos]
         box_coordinates = [int(biggest_area[0].item()), int(biggest_area[1].item()), int(biggest_area[2].item()), int(biggest_area[3].item())]

   else:
      box_coordinates = [0,0, img.shape[1], img.shape[0]]
   
   return box_coordinates

