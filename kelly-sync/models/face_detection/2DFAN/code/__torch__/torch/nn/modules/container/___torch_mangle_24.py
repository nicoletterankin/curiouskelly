class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  __annotations__["0"] = __torch__.torch.nn.modules.batchnorm.___torch_mangle_21.BatchNorm2d
  __annotations__["1"] = __torch__.torch.nn.modules.activation.___torch_mangle_22.ReLU
  __annotations__["2"] = __torch__.torch.nn.modules.conv.___torch_mangle_23.Conv2d
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_24.Sequential,
    argument_1: Tensor) -> Tensor:
    _0 = getattr(self, "2")
    _1 = getattr(self, "1")
    _2 = (getattr(self, "0")).forward(argument_1, )
    return (_0).forward((_1).forward(_2, ), )
