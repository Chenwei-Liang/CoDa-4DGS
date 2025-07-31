import json
import matplotlib.pyplot as plt

file_path = ''  
data = []

with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

steps = [int(entry['step']) for entry in data]
loss_origin = [float(entry['loss_origin']) for entry in data]
psnr = [float(entry['psnr']) for entry in data]


def calculate_ema(data, alpha=0.4):
    ema = [data[0]]  
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema

ema_loss_origin = calculate_ema(loss_origin)
ema_psnr = calculate_ema(psnr)

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(steps, loss_origin, label='Loss Origin', marker='o', color='blue')
ax1.plot(steps, ema_loss_origin, label='EMA Loss Origin', linestyle='--', color='blue')
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss Value')
ax1.set_title('Loss Components and PSNR with EMA vs. Step')
ax1.legend(loc='upper left')
ax1.grid()

ax2 = ax1.twinx()
ax2.plot(steps, ema_psnr, label='EMA PSNR', linestyle='--', color='purple')
ax2.set_ylabel('PSNR')
ax2.legend(loc='upper right')

plt.savefig("loss_plot_with_ema.png")
plt.show()
