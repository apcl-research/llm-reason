import time

#out_dir = 'out-owt-gpt2mini'
out_dir = 'out-jam-codetrans'
eval_interval = 5
eval_iters = 20
wandb_log = True
wandb_project = 'jam-codetrans'
wandb_run_name = 'jam-codetrans-base'

dataset = 'jam_codetrans'
init_from = 'resume'

# only save checkpoints if the validation loss improves
always_save_checkpoint = True 

# jam-cgpt 170k dataset has 37,399,419 tokens

# model iters
# 38m parameters model has 757,000 iters
# 110m parameters model has 762,000 iters
# 350m parameters model has 272,000 iters

block_size = 1024 

batch_size = 4 #16
gradient_accumulation_steps = 32


max_iters = 127000 + 15 * 3

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
