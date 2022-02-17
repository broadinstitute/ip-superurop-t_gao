#!/usr/bin/env python3

import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict



# set parameters
seed = 123
img_dir = pathlib.Path('Greyscale_Images_png/')
img_height = 256
img_width = 256
batch_size = 32
validation_split = 0.2
epoch_count = 100
verbose = True
num_classes = len(next(os.walk(img_dir))[1])
random.seed(seed)

# log hyperparameters to Neptune.ai
parameters = {
    'seed': seed,
    'img_height': img_height,
    'img_width': img_width,
    'batch_size': batch_size,
    'validation_split': validation_split,
    'n_epochs': epoch_count,
    'num_classes': num_classes
}



# implement ResNet (from https://github.com/FrancescoSaverioZuppichini/ResNet)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm2d(self.expanded_channels)

        })) if self.should_apply_shortcut else None


    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation=nn.ReLU, block=ResNetBasicBlock, *args,**kwargs):
        super().__init__()
        
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,  *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class ResNetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):
    
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResNetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def resnet18(in_channels, n_classes):
    return ResNet(in_channels, n_classes, block=ResNetBasicBlock, depths=[2, 2, 2, 2])



if __name__ == '__main__':
    
    # load datasets

    train_ds = utils.image_dataset_from_directory(
        img_dir,
        validation_split=0.2,
        subset='training',
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    ).cache().prefetch(buffer_size=AUTOTUNE)

    validation_ds = utils.image_dataset_from_directory(
        img_dir,
        validation_split=0.2,
        subset='validation',
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size
    ).cache().prefetch(buffer_size=AUTOTUNE)

    model = resnet18(in_channels=1, n_classes=n_classes)

    # create callbacks to save model
    checkpoint_dir = "./"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        if verbose:
            print('created checkpoint_dir', checkpoint_dir)
    cp_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_dir+ '_checkpoint_epoch-{epoch:0>3d}_loss-{val_loss:.2f}.hdf5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    earlystopping = callbacks.EarlyStopping(
        monitor ="val_loss",
        # mode ="min",
        patience = 5,
        restore_best_weights = True
    )

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epoch_count,
        callbacks=[NeptuneCallback(run=run), cp_callback, earlystopping]
    ).history

    # upload model files to Neptune
    run['model/saved_model'].upload_files('*.hdf5')
    run['model/graph'].upload_files('*.png')

    # plot loss and accuracy
    fig, axs = plt.subplots(2, 1, figsize=(15,15))
    axs[0].plot(history['loss'])
    axs[0].plot(history['val_loss'])
    axs[0].title.set_text('Training Loss vs Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend(['Train', 'Val'])
    axs[1].plot(history['accuracy'])
    axs[1].plot(history['val_accuracy'])
    axs[1].title.set_text('Training Accuracy vs. Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(['Train', 'Val'])
    plt.savefig('loss-accuracy.png', bbox_inches='tight')

    print('\ndone :)')
