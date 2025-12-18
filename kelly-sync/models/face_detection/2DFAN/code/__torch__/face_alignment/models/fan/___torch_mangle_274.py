class ConvBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  bn1 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_268.BatchNorm2d
  conv1 : __torch__.torch.nn.modules.conv.___torch_mangle_269.Conv2d
  bn2 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_270.BatchNorm2d
  conv2 : __torch__.torch.nn.modules.conv.___torch_mangle_271.Conv2d
  bn3 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_272.BatchNorm2d
  conv3 : __torch__.torch.nn.modules.conv.___torch_mangle_273.Conv2d
  def forward(self: __torch__.face_alignment.models.fan.___torch_mangle_274.ConvBlock,
    input: Tensor) -> Tensor:
    _0 = self.conv3
    _1 = self.bn3
    _2 = self.conv2
    _3 = self.bn2
    _4 = self.conv1
    input0 = torch.relu_((self.bn1).forward(input, ))
    _5 = (_4).forward(input0, )
    input1 = torch.relu_((_3).forward(_5, ))
    _6 = (_2).forward(input1, )
    input2 = torch.relu_((_1).forward(_6, ))
    out3 = torch.cat([_5, _6, (_0).forward(input2, )], 1)
    return torch.add_(out3, input, alpha=1)
