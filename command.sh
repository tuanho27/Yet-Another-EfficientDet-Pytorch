#Gen data
# mkdir datasets
# cd datasets; ln -s /mnt/fast_house/dataset/vinxray/train_images/ train; ln -s /mnt/fast_house/dataset/vinxray/train_images/ val
#  python voccoco.py --ann_dir /mnt/fast_house/dataset/vinxray/Annotations/ \
#                    --ann_paths_list /mnt/fast_house/dataset/vinxray/train_fold0.txt \
#                    --labels ./label.txt   \
#                    --output train_fold0.json

python train.py -c 3 -p vinxray --batch_size 4 --lr 1e-3 --num_epochs 50 \
                    --load_weights ./logs/vinxray/efficientdet-d3_19_17500.pth