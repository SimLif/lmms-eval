# conda activate lmms-eval
# bash eval.sh --model med_moe_phi --tasks medmoe_vqa --gpu 1
# bash eval.sh --model moe_llava_qwen --tasks vqa_rad
# bash eval.sh --model moe_llava_qwen --tasks medmoe_vqa
# bash eval.sh --model moe_llava_qwen_med --model_suffix 1epoch --tasks medmoe_vqa --gpu 1
# bash eval.sh --model moe_llava_qwen_med --model_suffix 9epoch --tasks medmoe_vqa --gpu 0
# bash eval.sh --model moe_llava_qwen_med --model_suffix s2_1epoch --tasks medmoe_vqa --gpu 0
# bash eval.sh --model moe_llava_qwen_med --model_suffix s2_9epoch --tasks medmoe_vqa --gpu 0
# bash eval.sh --model moe_llava_qwen_med --model_suffix s2_k_9epoch --tasks medmoe_vqa --gpu 0


# conda activate lmms-eval-twin
# bash eval.sh --model qwen2_vl --model_suffix instruct --tasks medmoe_vqa --gpu 0
# bash eval.sh --model qwen2_vl_adamllm --tasks medmoe_vqa --gpu 0
# bash eval.sh --model moe_qwen2_vl --model_suffix test --tasks medmoe_vqa --gpu 0 --port 29504

# bash eval.sh --model adamllm_qwen_med --tasks pmc_vqa --gpu 0 --port 29504
# bash eval.sh --model qwen2_vl --model_suffix instruct --tasks pmc_vqa --gpu 0 --port 29504


# bash eval.sh --model moe_llava_qwen --tasks pmc_vqa --gpu 1
# bash eval.sh --model med_moe_phi --tasks pmc_vqa --gpu 1
# bash eval.sh --model med_moe_stablelm --tasks pmc_vqa --gpu 1

# bash eval.sh --model moe_llava_qwen_med --model_suffix 1epoch --tasks pmc_vqa --gpu 1
# bash eval.sh --model moe_llava_qwen_med --model_suffix 9epoch --tasks pmc_vqa --gpu 1
# bash eval.sh --model moe_llava_qwen_med --model_suffix s2_1epoch --tasks pmc_vqa --gpu 1
# bash eval.sh --model moe_llava_qwen_med --model_suffix s2_9epoch --tasks pmc_vqa --gpu 1
# bash eval.sh --model moe_llava_qwen_med --model_suffix s2_k_9epoch --tasks pmc_vqa --gpu 1


# bash eval.sh --model moe_qwen2_vl --model_suffix ada_1epoch --tasks medmoe_vqa --gpu 0


# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_5epoch --tasks medmoe_vqa --gpu 1 --port 29501 2>&1 | tee output-1.log
# bash eval.sh --model moe_qwen2_vl --model_suffix 6e2_nano_ds_tok_share_5epoch --tasks medmoe_vqa --gpu 1 --port 29501 2>&1 | tee output-1.log
# bash eval.sh --model moe_qwen2_vl --model_suffix 12e4_nano_ds_tok_share_5epoch --tasks medmoe_vqa --gpu 1 --port 29501 2>&1 | tee output-1.log
# bash eval.sh --model moe_qwen2_vl --model_suffix 24e8_nano_ds_tok_share_5epoch --tasks medmoe_vqa --gpu 1 --port 29501 2>&1 | tee output-1.log
# bash eval.sh --model moe_qwen2_vl --model_suffix 48e16_nano_ds_tok_share_5epoch --tasks medmoe_vqa --gpu 1 --port 29501 2>&1 | tee output-1.log
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_nano_ds_tok_share_5epoch --tasks medmoe_vqa --gpu 1 --port 29501 2>&1 | tee output-1.log
# bash eval.sh --model moe_qwen2_vl --model_suffix 6e2_nano_ds_tok_share_r140_5epoch --tasks medmoe_vqa --gpu 1 --port 29501 2>&1 | tee output-1.log
# bash eval.sh --model moe_qwen2_vl --model_suffix 12e2_nano_ds_tok_share_r140_5epoch --tasks medmoe_vqa --gpu 1 --port 29501 2>&1 | tee output-1.log
# bash eval.sh --model moe_qwen2_vl --model_suffix 24e2_nano_ds_tok_share_r140_5epoch --tasks medmoe_vqa --gpu 1 --port 29501 2>&1 | tee output-1.log
# bash eval.sh --model moe_qwen2_vl --model_suffix 48e2_nano_ds_tok_share_r140_5epoch --tasks medmoe_vqa --gpu 1 --port 29501 2>&1 | tee output-1.log
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e2_nano_ds_tok_share_r140_5epoch --tasks medmoe_vqa --gpu 1 --port 29501 2>&1 | tee output-1.log

# bash eval.sh --model moe_qwen2_vl --model_suffix 192e32_nano_ds_tok_share_5epoch --tasks medmoe_vqa --gpu 5 --port 29505
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e16_nano_ds_tok_share_5epoch --tasks medmoe_vqa --gpu 5 --port 29505
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e8_nano_ds_tok_share_5epoch --tasks medmoe_vqa --gpu 5 --port 29505
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e4_nano_ds_tok_share_5epoch --tasks medmoe_vqa --gpu 5 --port 29505


# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_1000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_1500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_2000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_2500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_3000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_3500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_4000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_4500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_5000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_5500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_6000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_6500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_7000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_7500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_8000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch_8500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_share_10epoch --tasks medmoe_vqa --gpu 4 --port 29504

# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_500 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_1000 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_1500 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_2000 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_2500 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_3000 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_3500 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_4000 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_4500 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_5000 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_5500 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_6000 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_6500 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_7000 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_7500 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_8000 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_8500 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_share_10epoch_860 --tasks medmoe_vqa --gpu 2 --port 29502

# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_500 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_1000 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_1500 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_2000 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_2500 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_3000 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_3500 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_4000 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_4500 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_5000 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_5500 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_6000 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_6500 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_7000 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_7500 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_8000 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_8500 --tasks medmoe_vqa --gpu 7 --port 29507
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_ada_10epoch_8670 --tasks medmoe_vqa --gpu 7 --port 29507
# python modify_topk.py --topk 32
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e64_nano_ds_tok_share_5epoch --tasks medmoe_vqa --gpu 7 --port 29507
# python modify_topk.py --topk 16
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e64_nano_ds_tok_share_5epoch --tasks medmoe_vqa --gpu 7 --port 29507
# python modify_topk.py --topk 8
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e64_nano_ds_tok_share_5epoch --tasks medmoe_vqa --gpu 7 --port 29507
# python modify_topk.py --topk 4
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e64_nano_ds_tok_share_5epoch --tasks medmoe_vqa --gpu 7 --port 29507
# python modify_topk.py --topk 2
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e64_nano_ds_tok_share_5epoch --tasks medmoe_vqa --gpu 7 --port 29507

# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_1000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_1500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_2000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_2500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_3000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_3500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_4000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_4500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_5000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_5500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_6000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_6500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_7000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_7500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_8000 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_8500 --tasks medmoe_vqa --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_10epoch_8670 --tasks medmoe_vqa --gpu 4 --port 29504

# bash eval.sh --model qwen2_vl --model_suffix adamllm_med_full_5epoch --tasks text_char_substitution_005 --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 4e2_med_ada_5epoch --tasks text_char_substitution_005 --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 3e1_ada_med_5epoch --tasks text_char_substitution_005 --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 16e8_nano_ds_tok_ada_pre_5epoch --tasks text_char_substitution_005 --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 12e4_nano_ds_tok_share_ada_pre_5epoch --tasks text_char_substitution_005 --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_ada_med_5epoch --tasks text_char_substitution_005 --gpu 1 --port 29501

# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_ada_re_10epoch_500 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_ada_re_10epoch_1000 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_ada_re_10epoch_1500 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_ada_re_10epoch_2000 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_ada_re_10epoch_2500 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_ada_re_10epoch_3000 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_ada_re_10epoch_3500 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_ada_re_10epoch_4000 --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_ada_re_10epoch_4500 --tasks medmoe_vqa --gpu 2 --port 29502


# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_nano_ds_tok_share_5epoch --tasks text_proportional_char_substitution --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e64_nano_ds_tok_share_5epoch --tasks text_proportional_char_substitution --gpu 1 --port 29501

# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_mmed_med_5epoch --tasks image_gaussian_noise --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_mmed_med_5epoch --tasks image_rotation --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 12e4_mmed_med_5epoch --tasks text_char_substitution_005 --gpu 0 --port 29500
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_mmed_med_5epoch --tasks text_proportional_char_substitution --gpu 1 --port 29501


# bash eval.sh --model moe_qwen2_vl --model_suffix 192e64_nano_s_tok_share_5epoch --tasks medmoe_vqa --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e64_nano_s_tok_share_5epoch --tasks image_gaussian_noise --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e64_nano_s_tok_share_5epoch --tasks image_rotation --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e64_nano_s_tok_share_5epoch --tasks text_proportional_char_substitution --gpu 2 --port 29502


#  bash eval.sh --model moe_qwen2_vl --model_suffix 6e2_nano_s_tok_share_5epoch --tasks vqa_rad --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 12e4_nano_s_tok_share_5epoch --tasks vqa_rad --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 24e8_nano_s_tok_share_5epoch --tasks vqa_rad --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 48e16_nano_s_tok_share_5epoch --tasks vqa_rad --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_nano_s_tok_share_5epoch --tasks vqa_rad --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e64_nano_s_tok_share_5epoch --tasks vqa_rad --gpu 1 --port 29501

#  6e2_nano_ds_tok_share_5epoch
# bash eval.sh --model moe_qwen2_vl --model_suffix 6e2_nano_ds_tok_share_5epoch --tasks vqa_rad --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 12e4_nano_ds_tok_share_5epoch --tasks vqa_rad --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 24e8_nano_ds_tok_share_5epoch --tasks vqa_rad --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 48e16_nano_ds_tok_share_5epoch --tasks vqa_rad --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_nano_ds_tok_share_5epoch --tasks vqa_rad --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e64_nano_ds_tok_share_5epoch --tasks vqa_rad --gpu 1 --port 29501


# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_nano_ds_tok_share_5epoch --tasks image_rotation --gpu 0 --port 29500
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e64_nano_ds_tok_share_5epoch --tasks image_rotation --gpu 1 --port 29501
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_nano_ds_tok_share_5epoch --tasks image_gaussian_noise --gpu 2 --port 29502
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e64_nano_ds_tok_share_5epoch --tasks image_gaussian_noise --gpu 3 --port 29503
# bash eval.sh --model moe_qwen2_vl --model_suffix 96e32_nano_ds_tok_share_5epoch --tasks text_proportional_char_substitution --gpu 4 --port 29504
# bash eval.sh --model moe_qwen2_vl --model_suffix 192e64_nano_ds_tok_share_5epoch --tasks text_proportional_char_substitution --gpu 5 --port 29505

bash eval.sh --model qwen2_vl --model_suffix instruct  --tasks pmc_vqa,omni_med_vqa_mini --gpu 0 --port 29500
bash eval.sh --model adamllm_qwen_med  --tasks pmc_vqa,omni_med_vqa_mini --gpu 0 --port 29500