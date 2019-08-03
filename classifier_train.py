import torch
import torch.nn as nn
from datasets.classifier import classifier
from torch.utils.data import DataLoader
from backbones.hourglass_UNDER_flip import hourglass_net


MAX_ITER = 50000

train = classifier('./data/2018origin', mode='train_2019')
val = classifier('./data/2018origin', mode='new_val')
train_data = DataLoader(train, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_data = DataLoader(val, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)


def train():
    net = hourglass_net().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    net.train()
    epoch = 0
    train_loader = iter(train_data)

    total_loss = 0
    for step in range(1, MAX_ITER+1):
        try:
            data = next(train_loader)
        except:
            train_loader = iter(train_data)
            data = next(train_loader)
            epoch += 1

        optimizer.zero_grad()
        imgs = data[0].cuda()
        label = data[1].cuda().long().squeeze()
        name = data[2]

        output = net(imgs)
        try:
            loss = criterion(output[1], label)
        except:
            print(output[1].size(), label.size())
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        if step % 100 == 0:
            print('step:{}, loss:{}'.format(step, float(total_loss) / 100))
            total_loss = 0

        if step % 1000 == 0:
            print('start eval...')
            net.eval()
            step_eval = 0
            total_loss = 0
            correct = 0
            for data in val_data:
                imgs = data[0].cuda()
                label = data[1].cuda().long().squeeze()

                output = net(imgs)
                loss = criterion(output[1], label)

                # accuracy
                pred = output[1].data.max(1)[1]
                correct += float(pred.eq(label.data).sum())

                total_loss += loss.item()
                step_eval += 1

            avg_loss = total_loss / step_eval
            avg_acc = correct / (step_eval * 8)
            print('VAL avg_loss:{}, avg_acc:{}'.format(avg_loss, avg_acc))
            net.train()

        if step % 3000 == 0:
            torch.save(net.state_dict(), './hourglass_UNDER_{}.pth'.format(step))
    # torch.save(net.state_dict(), './hourglass_UNDER.pth')
    print('done!!')




if __name__ == '__main__':
    train()





