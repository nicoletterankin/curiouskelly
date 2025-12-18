class ConvBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  bn1 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_15.BatchNorm2d
  conv1 : __torch__.torch.nn.modules.conv.___torch_mangle_16.Conv2d
  bn2 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_17.BatchNorm2d
  conv2 : __torch__.torch.nn.modules.conv.___torch_mangle_18.Conv2d
  bn3 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_19.BatchNorm2d
  conv3 : __torch__.torch.nn.modules.conv.___torch_mangle_20.Conv2d
  downsample : __torch__.torch.nn.modules.container.___torch_mangle_24.Sequential
  def forward(self: __torch__.face_alignment.models.fan.___torch_mangle_25.ConvBlock,
    argument_1: Tensor) -> Tensor:
    _0 = self.downsample
    _1 = self.conv3
    _2 = self.bn3
    _3 = self.conv2
    _4 = self.bn2
    _5 = self.conv1
    input = torch.relu_((self.bn1).forward(argument_1, ))
    _6 = (_5).forward(input, )
    input0 = torch.relu_((_4).forward(_6, ))
    _7 = (_3).forward(input0, )
    input1 = torch.relu_((_2).forward(_7, ))
    out3 = torch.cat([_6, _7, (_1).forward(input1, )], 1)
    input2 = torch.add_(out3, (_0).forward(argument_1, ), alpha=1)
    return input2
