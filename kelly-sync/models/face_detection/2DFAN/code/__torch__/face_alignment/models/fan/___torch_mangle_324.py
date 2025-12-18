class HourGlass(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  b1_4 : __torch__.face_alignment.models.fan.___torch_mangle_239.ConvBlock
  b2_4 : __torch__.face_alignment.models.fan.___torch_mangle_246.ConvBlock
  b1_3 : __torch__.face_alignment.models.fan.___torch_mangle_253.ConvBlock
  b2_3 : __torch__.face_alignment.models.fan.___torch_mangle_260.ConvBlock
  b1_2 : __torch__.face_alignment.models.fan.___torch_mangle_267.ConvBlock
  b2_2 : __torch__.face_alignment.models.fan.___torch_mangle_274.ConvBlock
  b1_1 : __torch__.face_alignment.models.fan.___torch_mangle_281.ConvBlock
  b2_1 : __torch__.face_alignment.models.fan.___torch_mangle_288.ConvBlock
  b2_plus_1 : __torch__.face_alignment.models.fan.___torch_mangle_295.ConvBlock
  b3_1 : __torch__.face_alignment.models.fan.___torch_mangle_302.ConvBlock
  b3_2 : __torch__.face_alignment.models.fan.___torch_mangle_309.ConvBlock
  b3_3 : __torch__.face_alignment.models.fan.___torch_mangle_316.ConvBlock
  b3_4 : __torch__.face_alignment.models.fan.___torch_mangle_323.ConvBlock
  def forward(self: __torch__.face_alignment.models.fan.___torch_mangle_324.HourGlass,
    input: Tensor) -> Tensor:
    _0 = self.b3_4
    _1 = self.b3_3
    _2 = self.b3_2
    _3 = self.b3_1
    _4 = self.b2_plus_1
    _5 = self.b2_1
    _6 = self.b1_1
    _7 = self.b2_2
    _8 = self.b1_2
    _9 = self.b2_3
    _10 = self.b1_3
    _11 = self.b2_4
    _12 = (self.b1_4).forward(input, )
    input0 = torch.avg_pool2d(input, [2, 2], [2, 2], [0, 0], False, True, None)
    _13 = (_11).forward(input0, )
    _14 = (_10).forward(_13, )
    input1 = torch.avg_pool2d(_13, [2, 2], [2, 2], [0, 0], False, True, None)
    _15 = (_9).forward(input1, )
    _16 = (_8).forward(_15, )
    input2 = torch.avg_pool2d(_15, [2, 2], [2, 2], [0, 0], False, True, None)
    _17 = (_7).forward(input2, )
    _18 = (_6).forward(_17, )
    input3 = torch.avg_pool2d(_17, [2, 2], [2, 2], [0, 0], False, True, None)
    _19 = (_4).forward((_5).forward(input3, ), )
    up2 = torch.upsample_nearest2d((_3).forward(_19, ), None, [2., 2.])
    input4 = torch.add(_18, up2, alpha=1)
    up20 = torch.upsample_nearest2d((_2).forward(input4, ), None, [2., 2.])
    input5 = torch.add(_16, up20, alpha=1)
    up21 = torch.upsample_nearest2d((_1).forward(input5, ), None, [2., 2.])
    input6 = torch.add(_14, up21, alpha=1)
    up22 = torch.upsample_nearest2d((_0).forward(input6, ), None, [2., 2.])
    return torch.add(_12, up22, alpha=1)
