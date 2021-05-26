from Dataset_source import get_data
from utils.Dataset_seg2D import make_data_loaders
from Models.attention_unet import AttU_Net
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from utils.early_stopping import EarlyStopping
from utils import loss
from skimage import  transform


def total_loss(input, target, weights=[0.6, 0.2, 0.1, 0.1]):
    """
    	a list of input tesor of shape = (N, C, H, W)
    	target tensor of shape = (N, 1, H, W)
    """
    n = len(input)
    loss_dict = {}
    s = target.shape
    total_dice_loss = 0
    total_cross_entropy_loss = 0
    total_loss = 0
    target = target.cpu()

    for i in range(n):
        target_size = [s[0], s[1], int(s[2]//(2**i)), int(s[3]//(2**i))]
        rescale_target = transform.resize(target, target_size, order=0)
        rescale_target = torch.from_numpy(rescale_target)
        rescale_target = rescale_target.cuda(0)
        logits = input[i].cuda(0)

        dice_loss = loss.MulticlassDiceLoss()(logits, rescale_target, weights=[0.8, 0.2])
        cross_entropy_loss = loss.MulticlassEntropyLoss()(logits, rescale_target)

        loss_value = (dice_loss + cross_entropy_loss)*weights[i]
        total_cross_entropy_loss += cross_entropy_loss*weights[i]
        total_dice_loss += dice_loss*weights[i]
        total_loss += loss_value

    loss_dict['dice_loss'] = total_dice_loss
    loss_dict['cross_entropy_loss'] = total_cross_entropy_loss
    loss_dict['total_loss'] = total_loss
    return loss_dict


def add_scalar(writer, value_dict, n, epoch):
    for key, value in value_dict.items():
        writer.add_scalar(str(key), value/n, epoch)
    writer.flush()


def train_val(model, loaders, optimizer,  criterion, early_stopping, device, log_path, loss_list, n_epochs=300):
    train_writer = SummaryWriter(os.path.join(log_path, 'train', 'log'))
    eval_writer = SummaryWriter(os.path.join(log_path, 'eval', 'log'))

    for epoch in range(n_epochs):
        train_loss_dict = {}
        eval_loss_dict = {}
        for loss_key in loss_list:
            train_loss_dict[loss_key] = 0
            eval_loss_dict[loss_key] = 0

        for phase in ['train', 'eval']:
            loader = loaders[phase]
            for batch_id, (batch_x, batch_y) in enumerate(loader):
                batch_x, batch_y = batch_x.cuda(device[0]), batch_y.cuda(device[0])
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(batch_x)
                    loss_dict = criterion(output, batch_y)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss_dict['total_loss'].backward()
                    optimizer.step()
                    print('\nTrain: Epoch is : {} ({} / {}), '.format(epoch, batch_id, len(loader)))
                    for key in train_loss_dict.keys():
                        value_new = train_loss_dict[key] + loss_dict[key].data.item()
                        train_loss_dict[key] = value_new
                        print(str(key) + ' is :{:.4f}, '.format(loss_dict[key].data.item()), end='')
                else:
                    print('\nEval: Epoch is : {} ({} / {}), '.format(epoch, batch_id, len(loader)))
                    for key in eval_loss_dict.keys():
                        value_new = eval_loss_dict[key] + loss_dict[key].data.item()
                        eval_loss_dict[key] = value_new
                        print(str(key) + ' is :{:.4f}, '.format(loss_dict[key].data.item()), end='')

            if phase == 'train':
                add_scalar(train_writer, train_loss_dict, len(loader), epoch)

            if phase == 'eval':
                dice_loss = eval_loss_dict['total_loss']/len(loader)
                state = {}
                state['model'] = model.state_dict()
                state['optimizer'] = optimizer.state_dict()
                model_path = os.path.join(log_path, 'model')
                if (epoch + 1) % 20 == 0:
                    file_name = os.path.join(model_path, 'epoch' + str(epoch + 1) + '_model.pth')
                    torch.save(state, file_name)
                add_scalar(eval_writer, eval_loss_dict, len(loader), epoch)

                early_stopping(dice_loss, state, epoch, model_path)
        if early_stopping.early_stop:
            print('Early stopping')
            break

    train_writer.close()
    eval_writer.close()
    return model


if __name__ == '__main__':
    device_ids = [0]
    data_root = r'/home/wyd/PycharmProjects/ZWW_seg/data/Low'
    dict_path = r'/home/wyd/PycharmProjects/ZWW_seg/data/Low_index/dict_index'
    data_modes = ['qsm.npy']
    roi_modes = ['ROI.npy']
    input_shape = (128, 128)

    log_path = r'/home/wyd/PycharmProjects/ZWW_seg/log'
    os.makedirs(os.path.join(log_path, 'train', 'log'), exist_ok=True)
    os.makedirs(os.path.join(log_path, 'eval', 'log'), exist_ok=True)
    os.makedirs(os.path.join(log_path, 'model'), exist_ok=True)
    batch_size = 32
    in_ch = 3
    n_classes = 3

    loaders = make_data_loaders(data_root, data_modes, roi_modes, dict_path, input_shape, get_data, batch_size=batch_size)
    network = AttU_Net(img_ch=in_ch, output_ch=n_classes)

    TF = False
    if TF==True:
        tf_model_path = None
        pretrained_model = network.load_state_dict(torch.load(tf_model_path)['model'])
        for param in pretrained_model.parameters():
            param.requires_grad = True

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(network, device_ids=device_ids)
        model = model.cuda(device=device_ids[0])
    else:
        model = network.cuda(device=device_ids[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = total_loss

    loss_list = ['dice_loss', 'cross_entropy_loss', 'total_loss']
    early_stopping = EarlyStopping(patience=20, verbose=True)
    train_val(model, loaders, optimizer, criterion, early_stopping, device_ids, log_path, loss_list, n_epochs=300)

