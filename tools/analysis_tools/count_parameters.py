# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmcv import Config

from mmselfsup.models import build_algorithm
import hiddenlayer as hl
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Count model parameters')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    model = build_algorithm(cfg.model)
    transforms = [
        # Fold Conv, BN, RELU layers into one
        hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
        # Fold Conv, BN layers together
        hl.transforms.Fold("Conv > BatchNorm", "ConvBn"),
        # Fold bottleneck blocks
        hl.transforms.Fold("""
            ((ConvBnRelu > ConvBnRelu > ConvBn) | ConvBn) > Add > Relu
            """, "BottleneckBlock", "Bottleneck Block"),
        # Fold residual blocks
        hl.transforms.Fold("""ConvBnRelu > ConvBnRelu > ConvBn > Add > Relu""",
                        "ResBlock", "Residual Block"),
        # Fold bottleneck blocks
        hl.transforms.Fold("""
            ((ConvBnRelu > ConvBn) | ConvBn) > Add > Relu
            """, "BottleneckBlock", "Bottleneck Block"),
        # Fold residual blocks
        hl.transforms.Fold("""ConvBnRelu > ConvBn > Add > Relu""",
                        "ResBlock", "Residual Block"),
        # Fold repeated blocks
        hl.transforms.FoldDuplicates(),
    ]
    # hl.build_graph(model.encoder_q[0], torch.zeros([32, 3, 224, 224]), transforms=transforms).save(f'{args.config}_diagram.png', 'png')
    hl.build_graph(model, [torch.zeros([32, 3, 224, 224])]*2, transforms=transforms).save(f'{args.config}_diagram.png', 'png')

    num_params = sum(p.numel() for p in model.parameters()) / 1000000.
    num_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad) / 1000000.
    num_backbone_params = sum(p.numel()
                              for p in model.backbone.parameters()) / 1000000.
    num_backbone_grad_params = sum(p.numel()
                                   for p in model.backbone.parameters()
                                   if p.requires_grad) / 1000000.
    print(f'Number of backbone parameters: {num_backbone_params:.5g} M')
    print(f'Number of backbone parameters requiring grad: '
          f'{num_backbone_grad_params:.5g} M')
    print(f'Number of total parameters: {num_params:.5g} M')
    print(f'Number of total parameters requiring grad: '
          f'{num_grad_params:.5g} M')


if __name__ == '__main__':
    main()
