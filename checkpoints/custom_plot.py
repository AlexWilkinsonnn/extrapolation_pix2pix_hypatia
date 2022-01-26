import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.lines import Line2D

plt.rc('font', family='serif')

realB_1 = np.load("realB_valid0.npy")[0,112:212, 1700:2100]
fakeB_1 = np.load("fakeB_valid0.npy")[0,112:212, 1700:2100]
realA_1 = np.load("realA_valid0.npy")[0,112:212, 1700:2100]
realB_2 = np.load("realB_valid1.npy")[0]
fakeB_2 = np.load("fakeB_valid1.npy")[0]
realA_2 = np.load("realA_valid1.npy")[0]


fig, ax = plt.subplots(2, 3, figsize=(12,8))

ax[0,0].imshow(np.ma.masked_where(realA_1 == 0, realA_1).T, interpolation='none', cmap='viridis', aspect=0.25)
ax[0,0].set_title("Input", fontsize=16)

ax[0,1].imshow(fakeB_1.T, interpolation='none', cmap='seismic', aspect=0.25, vmin=-332, vmax=332) 
ax[0,1].set_title("Output", fontsize=16)

ax[0,2].imshow(realB_1.T, interpolation='none', cmap='seismic', aspect=0.25, vmin=-332, vmax=332)
ax[0,2].set_title("Ground Truth", fontsize=16)

ax[1,0].imshow(np.ma.masked_where(realA_2 == 0, realA_2).T, interpolation='none', cmap='viridis', aspect=0.25)

ax[1,1].imshow(fakeB_2.T, interpolation='none', cmap='viridis', aspect=0.25)#, vmin=-20, vmax=200) 

ax[1,2].imshow(realB_2.T, interpolation='none', cmap='viridis', aspect=0.25)#, vmin=-20, vmax=200)


# fig, ax = plt.subplots(3, 2, figsize=(8,12))

# ax[0,0].imshow(fakeB_1.T, interpolation='none', cmap='viridis', aspect='equal', vmin=0) 
# ax[0,0].set_title("Output", fontsize=16)

# ax[0,1].imshow(realB_1.T, interpolation='none', cmap='viridis', aspect='equal', vmin=0)
# ax[0,1].set_title("Ground Truth", fontsize=16)

# ax[1,0].imshow(fakeB_2.T, interpolation='none', cmap='viridis', aspect='equal', vmin=0)

# ax[1,1].imshow(realB_2.T, interpolation='none', cmap='viridis', aspect='equal', vmin=0)

# ax[2,0].imshow(fakeB_3.T, interpolation='none', cmap='viridis', aspect='equal', vmin=0)

# ax[2,1].imshow(realB_3.T, interpolation='none', cmap='viridis', aspect='equal', vmin=0)

#ax[2,1].text(145, 285, "DUNE Simulation\nPreliminary", fontweight='bold', c='darkgrey', alpha=1, fontsize=12)

for a in ax.reshape(-1): 
    a.set_axis_off()

fig.tight_layout()

# Has to be this way round so I won't need to reformat the poster.
#ax[0,0].text(125, 245, "DUNE Simulation\nPreliminary", fontweight='bold', c='darkgrey', alpha=1, fontsize=12)
#ax[0,1].text(125, 245, "DUNE Simulation\nPreliminary", fontweight='bold', c='darkgrey', alpha=1, fontsize=12)
#ax[1,0].text(145, 285, "DUNE Simulation\nPreliminary", fontweight='bold', c='darkgrey', alpha=1, fontsize=12)
#ax[1,1].text(145, 285, "DUNE Simulation\nPreliminary", fontweight='bold', c='darkgrey', alpha=1, fontsize=12)
#ax[2,0].text(145, 285, "DUNE Simulation\nPreliminary", fontweight='bold', c='darkgrey', alpha=1, fontsize=12)

# plt.subplots_adjust(wspace=0, hspace=0.01)
#plt.savefig("custom_pix2pix_example.pdf", bbox_inches='tight')
# plt.close()
plt.show()



input_dir = "depos_X_512_collection_scaled_5"
realA = np.load("realA_valid1.npy")[0, :, 1350:1650]
fakeB = np.load("fakeB_valid1.npy")[0, :, 1350:1650]
realB = np.load("realB_valid1.npy")[0, :, 1350:1650]

ch = (0, 0)
for idx, col in enumerate(realA):
    if np.abs(col).sum() > ch[1]:
        ch = (idx, np.abs(col).sum())
ch = ch[0]

tick_adc_true = realB[ch,:]
tick_adc_fake = fakeB[ch,:]
tick_charge_true = realA[ch,:]
ticks = np.arange(1350, 1650)

fig, ax = plt.subplots(figsize=(12,6))

ax.hist(ticks, bins=len(ticks), weights=tick_adc_true, histtype='step', label="Ground Truth (ADC)", linewidth=0.8, color='#E69F00')
ax.hist(ticks, bins=len(ticks), weights=tick_adc_fake, histtype='step', label="Output (ADC)", linewidth=0.8, color='#56B4E9')
ax.set_ylabel("ADC", fontsize=14)
ax.set_xlabel("Tick", fontsize=14)
ax.set_xlim(1350, 1650)

#ax.text(110, 950, "DUNE Simulation\nPreliminary", fontweight='bold', c='darkgrey', alpha=1, fontsize=12)

ax2 = ax.twinx()
ax2.hist(ticks, bins=len(ticks), weights=tick_charge_true, histtype='step', label="Input (charge)", linewidth=0.8, color='#009E73')
ax2.set_ylabel("Charge", fontsize=14)

ax_ylims = ax.axes.get_ylim()
ax_yratio = ax_ylims[0] / ax_ylims[1]
ax2_ylims = ax2.axes.get_ylim()
ax2_yratio = ax2_ylims[0] / ax2_ylims[1]
if ax_yratio < ax2_yratio:
    ax2.set_ylim(bottom = ax2_ylims[1]*ax_yratio)
else:
    ax.set_ylim(bottom = ax_ylims[1]*ax2_yratio)
        
# plt.title("Channel {} in Read Out Plane".format(ch), fontsize=14)

handles, labels = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
handles += handles2
labels += labels2
new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
plt.legend(handles=new_handles, labels=labels, prop={'size': 12}, frameon=False)

# plt.savefig("custom_pix2pix_trace_example_U.pdf", bbox_inches='tight')
plt.close()
#plt.show()


