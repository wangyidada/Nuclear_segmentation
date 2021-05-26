import torch
import torch.nn as nn
from Models.attention_unet import AttU_Net, AttU_Net_TF


def TFNet(TF_weights_file):
    pretrained_model = AttU_Net(img_ch=3, output_ch=3)
    pretrained_model.load_state_dict(torch.load(TF_weights_file)['model'])
    for param in pretrained_model.parameters():
        param.requires_grad = True
    pretrained_dict = pretrained_model.state_dict()

    new_model = AttU_Net_TF(img_ch=3, output_ch=4)
    new_model_dict = new_model.state_dict()
    pretrained_dict = {k : v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)
    return new_model


class TransferNet(nn.Module):
    def __init__(self, model, TF_weights_file, input_dim, output_dim):
        super(TransferNet, self).__init__()
        model.load_state_dict(torch.load(TF_weights_file)['model'])
        for param in model.parameters():
            param.requires_grad = True

        self.pre_layer1 = nn.Sequential(*list(model.children()))[:-4]
        self.pre_layer2 = nn.Sequential(*list(model.children()))[:-8]
        self.pre_layer3 = nn.Sequential(*list(model.children()))[:-11]
        self.last_out = nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, padding=0)

    def forword(self, x):
        x = self.pre_layer1(x)
        out = self.last_out1(x)
        return out


if __name__ == '__main__':
    tf_model_path = r'/home/wyd/PycharmProjects/ZWW_seg/Low_model/log/model/Epoch1_best_model.pth'
    # TF_model = TransferNet(model, tf_model_path, input_dim, out_dim)
    TFNet(tf_model_path)
