import re
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

log_file = 'train.log'  # 替换为你的实际日志文件路径

# 存储 loss
epoch_losses1 = defaultdict(list)
epoch_losses2 = defaultdict(list)

iter_steps = []
iter_loss1 = []
iter_loss2 = []
iter_total_loss = []

# 读取日志
with open(log_file, 'r') as f:
    for line in f:
        match = re.search(r'epoch\[(\d+)\] iter\[(\d+)\] loss: \[([\d.]+), ([\d.]+)\]', line)
        if match:
            epoch = int(match.group(1))
            it = int(match.group(2))
            loss1 = float(match.group(3))
            loss2 = float(match.group(4))
            total = loss1 + loss2

            # 记录 per-epoch loss
            epoch_losses1[epoch].append(loss1)
            epoch_losses2[epoch].append(loss2)

            # 记录每一次迭代的 loss
            global_step = epoch * 1000 + it  # 组合成唯一 step id
            iter_steps.append(global_step)
            iter_loss1.append(loss1)
            iter_loss2.append(loss2)
            iter_total_loss.append(total)

# ==== 打印每个 epoch 的平均值 ====
epochs = sorted(epoch_losses1.keys())
avg_loss1 = []
avg_loss2 = []
avg_total = []

print("📋 所有 Epoch 的 Loss:")
print(f"{'Epoch':>6} | {'Loss1':>10} | {'Loss2':>10} | {'TotalLoss':>12}")
print("-" * 45)
for e in epochs:
    l1 = sum(epoch_losses1[e]) / len(epoch_losses1[e])
    l2 = sum(epoch_losses2[e]) / len(epoch_losses2[e])
    total = l1 + l2
    avg_loss1.append(l1)
    avg_loss2.append(l2)
    avg_total.append(total)
    print(f"{e:>6} | {l1:10.6f} | {l2:10.6f} | {total:12.6f}")

# 最小值
min_total_loss = min(avg_total)
min_epoch = epochs[avg_total.index(min_total_loss)]
print(f"\n✅ 最小总损失出现在 epoch {min_epoch}，平均总损失值为 {min_total_loss:.6f}")

# # ==== 保存为 CSV ====
# csv_file = 'all_epoch_losses.csv'
# with open(csv_file, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['Epoch', 'Loss1', 'Loss2', 'TotalLoss'])
#     for e, l1, l2, total in zip(epochs, avg_loss1, avg_loss2, avg_total):
#         writer.writerow([e, l1, l2, total])
# print(f"📁 所有 epoch 的损失已保存为：{csv_file}")

# ==== 图1：每个 epoch 的平均 loss ====
plt.figure(figsize=(12, 6))
plt.plot(epochs, avg_loss1, label='Avg Loss 1')
plt.plot(epochs, avg_loss2, label='Avg Loss 2')
plt.plot(epochs, avg_total, label='Avg Total Loss', linestyle='--')
plt.axvline(min_epoch, color='red', linestyle=':', label=f'Min Loss Epoch {min_epoch}')
plt.scatter([min_epoch], [min_total_loss], color='red')

plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Average Loss per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('average_loss_per_epoch_with_min.png')
print("✅ 图像保存：average_loss_per_epoch_with_min.png")

# ==== 图2：每次迭代的 loss 变化 ====
plt.figure(figsize=(12, 6))
plt.plot(iter_steps, iter_loss1, label='Loss 1 (iter)', alpha=0.5)
plt.plot(iter_steps, iter_loss2, label='Loss 2 (iter)', alpha=0.5)
plt.plot(iter_steps, iter_total_loss, label='Total Loss (iter)', alpha=0.7)

plt.xlabel('Training Step (epoch * 1000 + iter)')
plt.ylabel('Loss')
plt.title('Loss Curve per Iteration')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_per_iteration.png')
print("✅ 图像保存：loss_per_iteration.png")
