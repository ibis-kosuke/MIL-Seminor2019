expt_dir='expt_train_val4'
python main.py --is_training=1 --expt_dir=$expt_dir --epoch=100
python main.py --is_training=0 --expt_dir=$expt_dir
