#Gen data
# mkdir datasets
# cd datasets; ln -s /mnt/fast_house/dataset/vinxray/train_images/ train; ln -s /mnt/fast_house/dataset/vinxray/train_images/ val
## prepare dataset
#  python voccoco.py --ann_dir /mnt/fast_house/dataset/vinxray/Annotations/ \
#                    --ann_paths_list /mnt/fast_house/dataset/vinxray/train_fold0.txt \
#                    --labels ./label.txt   \
#                    --output train_fold0.json

## training
python train.py -c 4 -p vinxray --batch_size 2 --lr 1e-3 --num_epochs 30 \
                    --load_weights pretrained_weights/efficientdet-d4.pth
                    # --load_weights ./logs/vinxray/efficientdet-d4_27_49000.pth
#                     --load_weights logs/vinxray/efficientdet-d4_26_47439.pth



## eval

# python eval.py -c 3 -p vinxray \
#                 --weights ./logs/vinxray/efficientdet-d3_49_43900.pth
            #    --weights ./logs/vinxray/efficientdet-d4_27_49000.pth

## testing and submit
# python test.py -c 3 --data_path /mnt/fast_house/dataset/vinxray/test_images/ \
#                --load_weights ./logs/vinxray/efficientdet-d3_49_43900.pth

# python test.py -c 4 --data_path /mnt/fast_house/dataset/vinxray/test_images/ \
#                --load_weights ./logs/vinxray/efficientdet-d4_27_49000.pth