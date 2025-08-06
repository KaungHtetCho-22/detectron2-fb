from detectron2.data.datasets import register_coco_instances

register_coco_instances(
    "leukemia_train", {}, 
    "../../../../datasets/detection/dataset/leukemia_train.json", 
    "../../../../datasets/detection/dataset/images/train"
)

register_coco_instances(
    "leukemia_val", {}, 
    "../../../../datasets/detection/dataset/leukemia_val.json", 
    "../../../../datasets/detection/dataset/images/val"
)
