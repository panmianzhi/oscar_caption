python test.py \
--data_dir data/vinvl_coco \
--do_test \
--do_eval \
--test_yaml data/vinvl_coco/test.yaml \
--per_gpu_eval_batch_size 64 \
--num_beams 5 \
--max_gen_length 20 \
--eval_model_dir ckpts/coco_captioning_base_scst/checkpoint-15-66405 # could be base or large models
