class Conv2d(Module):
  __parameters__ = ["weight", ]
  __buffers__ = []
  weight : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.conv.___torch_mangle_188.Conv2d,
    input: Tensor) -> Tensor:
    input0 = torch._convolution(input, self.weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, True, False, True, True)
    return input0
