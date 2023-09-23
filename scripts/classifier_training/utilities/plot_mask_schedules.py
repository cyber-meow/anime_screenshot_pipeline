import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

from scheduler import MasksSchedule

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

try:
    schedule = sys.argv[1]
except:
    schedule = 'sigmoid'

batch_size = 16
no_epochs = 100
steps_per_epoch = 20000
total_steps = no_epochs * steps_per_epoch

cdwu_percent = 0.1
cdwu_steps = int(total_steps*cdwu_percent)

max_text_seq_len = 16

# device=device, mask_schedule=args.mask_schedule,
#            masking_behavior=args.masking_behavior, tokenizer=args.tokenizer, vocab_size=args.vocab_size,
#            batch_size=args.batch_size, max_text_seq_len=args.max_text_seq_len,
#            warmup_steps=mask_wu_steps, cooldown_steps=mask_cd_steps, total_steps=total_steps, cycles=.5

mask_scheduler = MasksSchedule(torch.device('cpu'), schedule,
    'constant', 'wp', 30522, batch_size, max_text_seq_len, cdwu_steps, cdwu_steps, total_steps)

sample_captions = torch.tensor([[101, 1015, 2611, 2630, 2159, 2829, 2606, 1015, 1015, 2611, 5967, 9427, 2849, 10557, 2159, 102]], dtype=torch.int64)
sample_captions = sample_captions.repeat(batch_size, 1)

masked_percent_list = []

for step in range(total_steps):

    captions_updated, labels_text = mask_scheduler.ret_mask([step], sample_captions)
    
    text_len = captions_updated.shape[0] * captions_updated.shape[1]
    masked_text_len = torch.where(captions_updated==1, 1, 0).sum().item()
    masked_percent = ( masked_text_len / text_len) * 100
    masked_percent_list.append(masked_percent)
    
    if step % 5000 == 0:
        print(step/total_steps, text_len, masked_text_len, masked_percent)
        # print(step, sample_captions, captions_updated, labels_text)
    
print(len(masked_percent_list))
plt.plot(np.arange(total_steps), masked_percent_list)
plt.ylim([-1, 101])
plt.xlabel('Global step')
plt.ylabel('Tokens (text) masked (%)')
plt.title('Percentage of tokens masked as function of training progress')
plt.show()
