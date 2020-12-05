import os

from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

root = '/st2/hyewon/Trials/CS576/resblock'
data_dir = '/w11/hyewon/data/cifar10'

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation_type='relu', use_bn=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()

        if activation_type == 'relu':
            self.act = nn.ReLU(True)
        elif activation_type == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif activation_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation_type == 'tanh':
            self.act = nn.Tanh()
        elif activation_type == 'none':
            self.act = nn.Identity() 
        else:
            raise ValueError('Unknown activation_type !')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        output = self.act(x)

        return output

class ResBlockPlain(nn.Module):
    def __init__(self, in_channels, use_bn=False):
        super(ResBlockPlain, self).__init__()
        self.conv2d1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        if use_bn:
            self.bn = nn.BatchNorm2d(in_channels)
        else:
            self.bn = nn.Identity()
        self.relu = nn.ReLU(True)
        self.conv2d2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv2d1(x)
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv2d2(out)
        out += x
        output = self.relu(out)
        return output

class ResBlockBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels, use_bn=False):
        super(ResBlockBottleneck, self).__init__()
        self.conv2d1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0) # Note: you must erase this line
        if use_bn:
            self.bn1 = nn.BatchNorm2d(hidden_channels)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(True)
        self.conv2d2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d3 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, padding=0)
        if use_bn:
            self.bn2 = nn.BatchNorm2d(hidden_channels)
        else:
            self.bn2 = nn.Identity()        

    def forward(self, x):
        out = self.conv2d1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2d2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2d3(out)
        out += x
        output = self.relu(out)
        return output 


class MyNetwork(nn.Module):
    def __init__(self, nf, resblock_type='plain', num_resblocks=[1, 1, 1], use_bn=False):
        super(MyNetwork, self).__init__()
        """Initialize an entire network module components.

        Illustration: https://docs.google.com/drawings/d/1dN2RLaCpK5W61A9s2WhdOfZDuDBn6JtIJmWmIAIMgtg/edit?usp=sharing

        Instructions:
            1. Implement an algorithm that initializes necessary components as illustrated in the above link. 
            2. Initialized network components will be referred in `forward` method 
               for constructing the dynamic computational graph.

        Args:
            1. nf (int): Number of output channels for the first nn.Conv2d Module. An abbreviation for num_filter.
            2. resblock_type (str, optional): Type of ResBlocks to use. ('plain' | 'bottleneck'. default: 'plain')
            3. num_resblocks (list or tuple, optional): A list or tuple of length 3. 
               Each item at i-th index indicates the number of residual blocks at i-th Residual Layer.  
               (default: [1, 1, 1])
            4. use_bn (bool, optional): Whether to use batch normalization. (default: False)
        """
        ################################
        ## P3.1. Write your code here ##
        self.conv1 = nn.Conv2d(3, nf, 3, 1, 1)
        self.act = nn.ReLU(True)
        self.avgpool1 = nn.AvgPool2d(2,2)
        inter_layer = []
        for i in range(num_resblocks[0]):
          inter_layer.append(ResBlockPlain(nf,use_bn=use_bn) if resblock_type=='plain' else ResBlockBottleneck(nf,nf//2,use_bn=use_bn))
        self.resblock1 = nn.Sequential(*inter_layer)
        self.conv2 = nn.Conv2d(nf, nf*2, 3, 1, 1)
        self.avgpool2 = nn.AvgPool2d(2,2)
        inter_layer = []
        for i in range(num_resblocks[1]):
          inter_layer.append(ResBlockPlain(nf*2,use_bn=use_bn) if resblock_type=='plain' else ResBlockBottleneck(nf*2,nf,use_bn=use_bn))
        self.resblock2 = nn.Sequential(*inter_layer)
        self.conv3 = nn.Conv2d(nf*2, nf*4, 3, 1, 1)
        self.avgpool3 = nn.AvgPool2d(2,2)
        inter_layer = []
        for i in range(num_resblocks[2]):
          inter_layer.append(ResBlockPlain(nf*4,use_bn=use_bn) if resblock_type=='plain' else ResBlockBottleneck(nf*4,nf*2,use_bn=use_bn))
        self.resblock3 = nn.Sequential(*inter_layer)
        
        self.conv4 = nn.Conv2d(nf*4, nf*8, 3, 1, 1)
        self.avgpool4 = nn.AvgPool2d(2,2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(nf*8*2*2,256)
        self.fc2 = nn.Linear(256,10)

        self.loss = nn.CrossEntropyLoss()
        self.apply(self.init_params)

    def forward(self, x):

        output = self.conv1(x)
        output = self.act(output)
        output = self.avgpool1(output)
        output = self.resblock1(output)
        output = self.conv2(output)
        output = self.act(output)
        output = self.avgpool2(output)
        output = self.resblock2(output)
        output = self.conv3(output)
        output = self.act(output)
        output = self.avgpool3(output)
        output = self.resblock3(output)
        output = self.conv4(output)
        output = self.act(output)
        output = self.avgpool4(output)
        output = self.flatten(output)
        output = self.fc1(output)
        output = self.act(output)
        output = self.fc2(output)

        return output

    def init_params(self, m):
        if isinstance(m, (nn.BatchNorm2d)):
          m.weight.data.fill_(1.0)
          m.bias.data.zero_()
        elif isinstance(m,(nn.Conv2d)):
          nn.init.kaiming_normal_(m.weight)
          m.bias.data.zero_()

        elif isinstance(m,(nn.Linear)):
          nn.init.kaiming_normal_(m.weight)
          m.bias.data.zero_()

    def compute_loss(self, logit, y):
        loss = self.loss(logit, y)
        return loss


class CIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(CIFAR10, self).__init__()
        self.transform = transform 
        self.root = root
        if train:
            self.data_dir = '/w11/hyewon/data/cifar10/train'
        else:
            self.data_dir = '/w11/hyewon/data/cifar10/test'
        
        self.paths = list(Path(self.data_dir).glob('**/*.png'))
        num_paths = len(self.paths)
        assert isinstance(self.paths, (list,)), 'Wrong type. self.paths should be list.'
        if train is True:
            assert len(self.paths) == 48000, 'There are 48,000 train images, but you have gathered %d image paths' % len(self.paths)
        else:
            assert len(self.paths) == 12000, 'There are 12,000 test images, but you have gathered %d image paths' % len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx] 
        path_name = str(path).split('/')
        # print (path_name)
        lbl = int(path_name[-2])
        label = torch.tensor(lbl).type(torch.LongTensor) # Note: you must erase this line

        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image) 

        return image, label

    def __len__(self):
        return len(self.paths)


# Check and test your CIFAR10 Dataset class here.
train = True
transform = transforms.ToTensor()

dset = CIFAR10(data_dir, train, transform)
print('num data:', len(dset))

x_test, y_test = dset[0]
print('image shape:', x_test.shape, '| type:', x_test.dtype)
print('label shape:', y_test.shape, '| type:', y_test.dtype)

def get_dataloader(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    train_dataset = CIFAR10(args.dataroot, train=True, transform=transform)
    test_dataset = CIFAR10(args.dataroot, train=False, transform=transform)

    # P4.4. Use `DataLoader` module for mini-batching train and test datasets.
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, test_dataloader

# Configurations & Hyper-parameters

from easydict import EasyDict as edict

args = edict()

# basic options 
args.name = 'main_plain_0.001_filter64'                   # experiment name.
args.ckpt_dir = 'ckpts'              # checkpoint directory name.
args.ckpt_iter = 100                 # how frequently checkpoints are saved.
args.ckpt_reload = 'best'            # which checkpoint to re-load.
args.gpu = True                      # whether or not to use gpu. 

# network options
args.num_filters = 64                # number of output channels in the first nn.Conv2d module in MyNetwork.
args.resblock_type = 'plain'    # type of residual block. ('plain' | 'bottleneck').
args.num_resblocks = [1, 2, 3]       # number of residual blocks in each Residual Layer.
args.use_bn = False                  # whether or not to use batch normalization.

# data options
args.dataroot = 'dataset/cifar10'    # where CIFAR10 images exist.
args.batch_size = 64                 # number of mini-batch size.

# training options
args.lr = 0.001                     # learning rate.
args.epoch = 50                      # training epoch.

# tensorboard options
args.tensorboard = True             # whether or not to use tensorboard logging.
args.log_dir = 'logs'                # to which tensorboard logs will be saved.
args.log_iter = 100                  # how frequently logs are saved.

# Basic settings
device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'

result_dir = Path(root) / 'results' /args.name
ckpt_dir = result_dir / args.ckpt_dir
ckpt_dir.mkdir(parents=True, exist_ok=True)
log_dir = result_dir / args.log_dir
log_dir.mkdir(parents=True, exist_ok=True)

global_step = 0
best_accuracy = 0.

# Setup tensorboard.
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter 
    writer = SummaryWriter(log_dir)
else:
    writer = None

# Define your model and optimizer
# Complete ResBlockPlain, ResBlockBottleneck, and MyNetwork modules to proceed further.
net = MyNetwork(args.num_filters, args.resblock_type, args.num_resblocks, args.use_bn).to(device)
optimizer = optim.Adam(net.parameters(), lr=args.lr)

# print(net)

# Re-load pre-trained weights if exists.
ckpt_path = ckpt_dir / ('%s.pt' % args.ckpt_reload)
try:
    net.load_state_dict(torch.load(ckpt_path))
except Exception as e:
    print(e)
# Get train/test data loaders  
# Complete CIFAR10 dataset class and get_dataloader method to proceed further.
train_dataloader, test_dataloader = get_dataloader(args)

for epoch in tqdm(range(args.epoch)):
    # Here starts the train loop.
    net.train()
    for x, y in train_dataloader:
        global_step += 1

        # P5.1. Send `x` and `y` to either cpu or gpu using `device` variable. 
        x = x.to(device) # x = write your code here (one-liner). 
        y = y.to(device) # y = write your code here (one-liner).
        
        # P5.2. Feed `x` into the network, get an output, and keep it in a variable called `logit`. 
        logit = net(x) # logit = write your code here (one-liner).

        # P5.3. Compute loss using `logit` and `y`, and keep it in a variable called `loss` 
        loss = net.compute_loss(logit, y) # loss =  write your code here (one-liner).
        accuracy = (logit.argmax(dim=1)==y).float().mean()

        # P5.4. flush out the previously computed gradient 
        optimizer.zero_grad() # write your code here (one-liner).

        # P5.5. backward the computed loss. 
        loss.backward() # write your code here (one-liner).

        # P5.6. update the network weights. 
        optimizer.step() # write your code here (one-liner).

        if global_step % args.log_iter == 0 and writer is not None:
            # P5.7. Log `loss` with a tag name 'train_loss' using `writer`. Use `global_step` as a timestamp for the log. 
            writer.add_scalar('train_loss', loss, global_step=global_step) # writer.writer_your_code_here (one-liner).
            # P5.8. Log `accuracy` with a tag name 'train_accuracy' using `writer`. Use `global_step` as a timestamp for the log. 
            writer.add_scalar('train_accuracy', accuracy, global_step=global_step) # writer.writer_your_code_here (one-liner).

        if global_step % args.ckpt_iter == 0: 
            # P5.9. Save network weights in the directory specified by `ckpt_dir` directory.
            #    Use `global_step` to specify the timestamp in the checkpoint filename.
            #    E.g) if `global_step=100`, the filename can be `100.pt`
            train_ckpt = str(ckpt_dir)+'/'+str(global_step)+'.pt'
            torch.save(net.state_dict(),train_ckpt) # write your code here (one-liner).


    # Here starts the test loop.
    net.eval()
    with torch.no_grad():
        test_loss = 0.
        test_accuracy = 0.
        test_num_data = 0.
        for x, y in test_dataloader:
            # P5.10. Send `x` and `y` to either cpu or gpu using `device` variable.
            x = x.to(device) # x = write your code here (one-liner).
            y = y.to(device) # y = write your code here (one-liner).

            # P5.11. Feed `x` into the network, get an output, and keep it in a variable called `logit`.
            logit = net(x) # logit = write your code here (one-liner). 

            # P5.12. Compute loss using `logit` and `y`, and keep it in a variable called `loss`
            loss = net.compute_loss(logit, y) # loss = write your code yere (one-liner). 
            accuracy = (logit.argmax(dim=1) == y).float().mean()

            test_loss += loss.item()*x.shape[0]
            test_accuracy += accuracy.item()*x.shape[0]
            test_num_data += x.shape[0]

        test_loss /= test_num_data
        test_accuracy /= test_num_data

        if writer is not None: 
            # P5.13. Log `test_loss` with a tag name 'test_loss' using `writer`. Use `global_step` as a timestamp for the log.
            writer.add_scalar('test_loss', test_loss, global_step) # writer.write_your_code_here (one-liner).
            # P5.14. Log `test_accuracy` with a tag name 'test_accuracy' using `writer`. Use `global_step` as a timestamp for the log.
            writer.add_scalar('test_accuracy', test_accuracy, global_step=global_step) # writer.write_your_code_here (one-liner).
            writer.flush()

        # P5.15. Whenever `test_accuracy` is greater than `best_accuracy`, save network weights with the filename 'best.pt' in the directory specified by `ckpt_dir`.
        #     Also, don't forget to update the `best_accuracy` properly.
        # write your code here. 
        if test_accuracy > best_accuracy:
            test_ckpt = str(ckpt_dir)+'/'+'best.pt'
            torch.save(net.state_dict(),test_ckpt)
            best_accuracy = test_accuracy