CUDA_VISIBLE_DEVICES=3 nohup python3 run_summarization.py --mode=train --vocab_path=data/vocab.txt --log_root=log_autoencoder --exp_name=myexperiment --gpuid=2 > log.txt &
