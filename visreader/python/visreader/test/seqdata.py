import os

#
# get absolute file path for different data
#
# maybe you should first download the files from here:
# https://onedrive.live.com/?authkey=%21AJZJXHcZyLQAh8E&id=910128F56B9FFE8D%21106&cid=910128F56B9FFE8D
data_root = os.path.dirname(os.path.realpath(__file__))
data_root = os.path.join(data_root, '../../../tests/data/seqfile')

imagenet_train = os.path.join(data_root, 'imagenet.train.seqfile')
imagenet_val = os.path.join(data_root, 'imagenet.val.seqfile')
coco_train = os.path.join(data_root, 'coco_train.seqfile')
coco_val = os.path.join(data_root, 'coco_val.seqfile')
