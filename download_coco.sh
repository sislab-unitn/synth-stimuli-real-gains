# Run from the repo root. Produces coco/val2017 and coco/annotations.
mkdir -p coco

# val data
wget http://images.cocodataset.org/zips/val2017.zip -P coco
unzip coco/val2017.zip -d coco

# annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P coco
unzip coco/annotations_trainval2017.zip -d coco

rm coco/val2017.zip coco/annotations_trainval2017.zip
