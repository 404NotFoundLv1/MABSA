. ./config/baseline_paths.sh

for sl in  '7e-5' #You can change the sl to find the best hyperparameter.
do
		echo ${sl}
		python MAESC_training.py \
          --dataset twitter15 ./src/data/jsons/twitter15_info.json \
          --checkpoint_dir ./ \
          --model_config config/pretrain_base.json \
          --log_dir 15_aesc \
          --num_beams 4 \
          --eval_every 1 \
          --lr ${sl} \
          --batch_size 16  \
          --epochs 35 \
          --grad_clip 5 \
          --warmup 0.1 \
          --seed 66 \
          --bart_model "${VLP_BART_MODEL}" \
          --checkpoint "${VLP_CHECKPOINT}"
done
