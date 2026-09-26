CUDA_VISIBLE_DEVICES=1 python3 run_single_model.py --dataset gsm8k 2>&1 | tee ~/workspace/tmp/recmas/logs/single_gsm8k.log
CUDA_VISIBLE_DEVICES=1 python3 run_text_mas.py --dataset gsm8k 2>&1 | tee ~/workspace/tmp/recmas/logs/textmas_gsm8k.log
CUDA_VISIBLE_DEVICES=1 python3 run_recursive_mas.py --dataset gsm8k 2>&1 | tee ~/workspace/tmp/recmas/logs/recmas_gsm8k.log
