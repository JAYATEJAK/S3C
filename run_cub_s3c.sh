Method=both_stochastic_self_sup_aggregation_expts_cub
mkdir -p cub_log_dir/${Method}/

python train.py -project s3c -dataset cub200  -base_mode 'ft_cos' \
-new_mode 'avg_cos' -gamma 0.1 -lr_base 0.03 -lr_new 0.003 -decay 0.0005 \
-epochs_base 100 -epochs_new 100 -schedule Milestone -milestones 30 40 60 80 \
-lamda_proto 5 \
-gpu 0 -temperature 16 --Method ${Method} \
2>&1 | tee ./cub_log_dir/${Method}/log_cub.txt
