from utils import isDebug
train_steps= 320000000
batch_size= 512
lr= 0.00001
memory_size= 1000000
init_fill = 20000
gamma= 0.99
target_entropy_ratio= 0.9
update_interval= 4   # collect 4 steps and train once
target_update_interval= 5000 # update target network every N trains, infected by update_interval
num_eval_steps= 50      #unit : every N train steps
log_interval= 300       #unit : every N train steps
eval_interval= 2000     #unit : every N train steps
save_interval=2000      #unit : every N train steps
dueling_net= True
use_PER= True
if isDebug():
    init_fill=1000