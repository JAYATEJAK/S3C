Method=both_stochastic_self_sup_aggregation_expts_cifar
Method=test

mkdir -p cifar_log_dir/${Method}/

python train.py -project s3c -dataset cifar100  -base_mode 'ft_cos' \
-new_mode 'avg_cos' -gamma 0.1 -lr_base 0.1 -lr_new 0.01 -decay 0.0005 \
-epochs_base 200 -epochs_new 100 -schedule Milestone -milestones 120 160 \
-lamda_proto 5 \
-gpu 0 -temperature 16 --Method ${Method} \
2>&1 | tee ./cifar_log_dir/${Method}/log_lamda_ce_1_lamda_proto_5.txt