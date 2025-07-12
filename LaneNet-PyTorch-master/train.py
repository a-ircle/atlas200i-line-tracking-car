import os
import time
import torch
from dataset.dataset_utils import TUSIMPLE
from Lanenet.model2 import Lanenet
from Lanenet.cluster_loss3 import cluster_loss

def main():
    # 数据路径，按需修改
    root = '/root/car_data_new' 

    # 数据集
    train_set = TUSIMPLE(root=root, flag='train')
    valid_set = TUSIMPLE(root=root, flag='valid')
    test_set = TUSIMPLE(root=root, flag='test')

    print('train_set length {}'.format(len(train_set)))
    print('valid_set length {}'.format(len(valid_set)))
    print('test_set length {}'.format(len(test_set)))

    # 超参数
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 10000

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 模型
    LaneNet_model = Lanenet(2, 4)
    LaneNet_model.to(device)

    # 预训练权重路径，按需修改
    pretrained_weights_path = '/root/LaneNet-Pytorch-teach/LaneNet-PyTorch-master/TUSIMPLE/Lanenet_output/lanenet_my_improve_84_batch_8.model'
    if os.path.exists(pretrained_weights_path):
        pretrained_weights = torch.load(pretrained_weights_path, map_location=device)
        LaneNet_model.load_state_dict(pretrained_weights, strict=False)
        print("Loaded pretrained weights.")
    else:
        print("Pretrained weights not found, training from scratch.")

    # 数据加载器
    data_loader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    data_loader_valid = torch.utils.data.DataLoader(valid_set, batch_size=1, shuffle=True, num_workers=2)
    data_loader_test = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    # 优化器和学习率调度
    params = [p for p in LaneNet_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0002)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 损失函数
    criterion = cluster_loss()

    loss_all = []

    for epoch in range(85, num_epochs):
        LaneNet_model.train()
        ts = time.time()
        for iter, batch in enumerate(data_loader_train):
            input_image = batch[0].to(device)
            binary_labels = batch[1].to(device)
            instance_labels = batch[2].to(device)

            binary_final_logits, instance_embedding = LaneNet_model(input_image)
            binary_seg_loss, instance_seg_loss = criterion(binary_logits=binary_final_logits, binary_labels=binary_labels,
                                                          instance_logits=instance_embedding, instance_labels=instance_labels,
                                                          delta_v=0.5, delta_d=3)

            loss = binary_seg_loss + instance_seg_loss

            optimizer.zero_grad()
            loss_all.append(loss.item())
            loss.backward()
            optimizer.step()

            if iter % 20 == 0:
                print(f"epoch[{epoch}] iter[{iter}] loss: [{binary_seg_loss.item():.4f}, {instance_seg_loss.item():.4f}]")

        lr_scheduler.step()
        print(f"Finish epoch[{epoch}], time elapsed[{time.time() - ts:.2f}s]")

        # 保存模型，路径和命名可修改
        save_dir = "TUSIMPLE/Lanenet_output"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(LaneNet_model.state_dict(), os.path.join(save_dir, f"lanenet_my_improve_{epoch}_batch_{batch_size}.model"))

if __name__ == "__main__":
    main()
