mkdir -p datasets
hf download fcakyon/pokemon-classification --repo-type dataset --local-dir ./datasets/pokemon-classification
cd ./datasets/pokemon-classification/data
unzip train.zip -d train
unzip test.zip -d  test
unzip valid.zip -d valid

# place hf auth token in .env in variable HF_TOKEN
source .env

# nohup hf download jameelkhalidawan/Plant_Detection_Classification --repo-type dataset --local-dir ./datasets/plant-classification --token $HF_TOKEN --max-workers 50 > plant-classification.log 2>&1 &
# 491747
# nohup hf download ILSVRC/imagenet-1k --repo-type dataset --local-dir ./datasets/imagenet-1k --token $HF_TOKEN --include="data/train_images_0.tar.gz" --max-workers 50 > imagenet-1k.log  2>&1 &
