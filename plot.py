import matplotlib.pyplot as plt
import os
import re
import pickle
import torch
import numpy as np

# deal with gpu tensors on cpu
torch.serialization.register_package(0, lambda x: x.device.type, lambda x, _: x.cpu())

# run in directory of pickled loss 
files = os.listdir()
losses = [l for l in files if l[-8:] == "loss.pkl"]

re_str = re.compile(r"beta_*(\d+\.\d+)_latentdim_(\d+)_loss\.pkl")

fig, axes = plt.subplots(2,1)

#https://stackoverflow.com/questions/1168260/algorithm-for-generating-unique-colors
colors=["#000000","#00FF00","#0000FF","#FF0000","#01FFFE","#FFA6FE","#FFDB66","#006401","#010067","#95003A","#007DB5","#FF00F6","#FFEEE8","#774D00","#90FB92","#0076FF","#D5FF00","#FF937E","#6A826C","#FF029D","#FE8900","#7A4782","#7E2DD2","#85A900","#FF0056","#A42400","#00AE7E","#683D3B","#BDC6FF","#263400","#BDD393","#00B917","#9E008E","#001544","#C28C9F","#FF74A3","#01D0FF","#004754","#E56FFE","#788231","#0E4CA1","#91D0CB","#BE9970","#968AE8","#BB8800","#43002C","#DEFF74","#00FFC6","#FFE502","#620E00","#008F9C","#98FF52","#7544B1","#B500FF","#00FF78","#FF6E41","#005F39","#6B6882","#5FAD4E","#A75740","#A5FFD2","#FFB167","#009BFF","#E85EBE"]


i = 0
for l in losses:
    match = re.match(re_str, l)
    beta = match.groups()[0]
    latent_dim = match.groups()[1]

    with open(l, 'rb') as f:
        data = pickle.load(f)
        tr_loss = data['tr_loss']
        te_loss = data['te_loss']
        tr_loss_arr=[tr_loss[i].detach().numpy() for i in range(len(tr_loss))]
        te_loss_arr=[te_loss[i].detach().numpy() for i in range(len(te_loss))]
        axes[0].plot(tr_loss_arr[:25], color=colors[i+6], label=f"({beta},{latent_dim})")
        axes[1].plot(te_loss_arr[:25], color=colors[i+6])

    i+=1

axes[0].set_ylabel('Training Loss')
axes[1].set_ylabel('Test Loss')
fig.legend(title=r"$\beta$, latent dim")
plt.savefig('bvae.pdf')
plt.show()
