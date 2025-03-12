import time

#out_dir = 'out-owt-gpt2mini'
out_dir = 'out-srcml-codegen'
eval_interval = 100
eval_iters = 80
wandb_log = True
wandb_project = 'jam_codegen'
wandb_run_name = 'jam-codegen-srcml'

dataset = 'jam_codegen'
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


max_iters = 570000 + 240 * 3

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
