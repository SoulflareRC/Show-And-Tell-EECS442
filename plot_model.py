import logging

import matplotlib.pyplot as plt
from pathlib import Path
from utils import restore_checkpoint,restore_checkpoint_from_file
import logging
logging.basicConfig()
logging.root.setLevel(logging.INFO)
import json


batch_size = 256
emb_dim = 1024
hidden_size = 512
log_dir = Path("log")
epochs = 10
log_period = 1
save_period = 1
name = "NIC"
config_name = "emb_1024_hidden_512_Adam_0.0001"
# config_name = f"emb_{emb_dim}_hidden_{hidden_size}"

# state = restore_checkpoint(log_dir,name,config_name)
state = restore_checkpoint_from_file("log/NICdropout_bn_emb_1024_hidden_512_Adam_0.0001/NIC_epoch_76_dropout_bn_emb_1024_hidden_512_Adam_0.0001.pth")
stats = state['stats']
stat = stats[-1]
metrics = state['metrics']
epochs = [s['epoch'] for s in stats]
values = ["train_loss","val_loss","train_acc","val_acc"]

logging.info(f"Loss & Acc:{json.dumps(stat,indent=4)}")
logging.info(f"Metrics:{json.dumps(metrics,indent=4)}")
# plt.title(f"{name}_{config_name}")
plt.title(f"NIC")
plt.xlabel(f"Epochs")
for val in values:
    vals = [s[val] for s in stats]
    plt.plot(epochs,vals,label=val)
plt.legend(loc="best")
plt.show()