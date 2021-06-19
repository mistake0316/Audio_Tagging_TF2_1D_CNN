# 目前這個版本有點問題，之後也許會修正，不過 pretrain weight現在可以使用
python3 Training_with_outer_loader.py \
  --model_folder="MULTI_CLASS_VERSION"\
  --final_group_mode="attention,last"\
  --multilabel_problem=0\
  --mixup=0