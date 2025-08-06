import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances

register_coco_instances("leukemia_train", {}, "leukemia_train.json", "../../../../leukemia/dataset/images/train")
register_coco_instances("leukemia_val", {}, "leukemia_val.json", "../../../../leukemia/dataset/images/val")

cfg = get_cfg()
cfg.merge_from_file("configs/COCO/leukemia-patch8-vit.yaml")
cfg.OUTPUT_DIR = "./output/leukemia_vitdet_small"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
