class ConvBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  bn1 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_157.BatchNorm2d
  conv1 : __torch__.torch.nn.modules.conv.___torch_mangle_158.Conv2d
  bn2 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_159.BatchNorm2d
  conv2 : __torch__.torch.nn.modules.conv.___torch_mangle_160.Conv2d
  bn3 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_161.BatchNorm2d
  conv3 : __torch__.torch.nn.modules.conv.___torch_mangle_162.Conv2d
  def forward(self: __torch__.face_alignment.models.fan.___torch_mangle_163.ConvBlock,
    argument_1: Tensor) -> Tensor:
    _0 = self.conv3
    _1 = self.bn3
    _2 = self.conv2
    _3 = self.bn2
    _4 = self.conv1
    input = torch.relu_((self.bn1).forward(argument_1, ))
    _5 = (_4).forward(input, )
    input0 = torch.relu_((_3).forward(_5, ))
    _6 = (_2).forward(input0, )
    input1 = torch.relu_((_1).forward(_6, ))
    out3 = torch.cat([_5, _6, (_0).forward(input1, )], 1)
    up1 = torch.add_(out3, argument_1, alpha=1)
    return up1
