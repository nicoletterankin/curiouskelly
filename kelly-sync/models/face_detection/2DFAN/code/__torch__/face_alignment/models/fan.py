class FAN(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  conv1 : __torch__.torch.nn.modules.conv.Conv2d
  bn1 : __torch__.torch.nn.modules.batchnorm.BatchNorm2d
  conv2 : __torch__.face_alignment.models.fan.ConvBlock
  conv3 : __torch__.face_alignment.models.fan.___torch_mangle_14.ConvBlock
  conv4 : __torch__.face_alignment.models.fan.___torch_mangle_25.ConvBlock
  m0 : __torch__.face_alignment.models.fan.HourGlass
  top_m_0 : __torch__.face_alignment.models.fan.___torch_mangle_123.ConvBlock
  conv_last0 : __torch__.torch.nn.modules.conv.___torch_mangle_124.Conv2d
  bn_end0 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_125.BatchNorm2d
  l0 : __torch__.torch.nn.modules.conv.___torch_mangle_126.Conv2d
  bl0 : __torch__.torch.nn.modules.conv.___torch_mangle_127.Conv2d
  al0 : __torch__.torch.nn.modules.conv.___torch_mangle_128.Conv2d
  m1 : __torch__.face_alignment.models.fan.___torch_mangle_220.HourGlass
  top_m_1 : __torch__.face_alignment.models.fan.___torch_mangle_227.ConvBlock
  conv_last1 : __torch__.torch.nn.modules.conv.___torch_mangle_228.Conv2d
  bn_end1 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_229.BatchNorm2d
  l1 : __torch__.torch.nn.modules.conv.___torch_mangle_230.Conv2d
  bl1 : __torch__.torch.nn.modules.conv.___torch_mangle_231.Conv2d
  al1 : __torch__.torch.nn.modules.conv.___torch_mangle_232.Conv2d
  m2 : __torch__.face_alignment.models.fan.___torch_mangle_324.HourGlass
  top_m_2 : __torch__.face_alignment.models.fan.___torch_mangle_331.ConvBlock
  conv_last2 : __torch__.torch.nn.modules.conv.___torch_mangle_332.Conv2d
  bn_end2 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_333.BatchNorm2d
  l2 : __torch__.torch.nn.modules.conv.___torch_mangle_334.Conv2d
  bl2 : __torch__.torch.nn.modules.conv.___torch_mangle_335.Conv2d
  al2 : __torch__.torch.nn.modules.conv.___torch_mangle_336.Conv2d
  m3 : __torch__.face_alignment.models.fan.___torch_mangle_428.HourGlass
  top_m_3 : __torch__.face_alignment.models.fan.___torch_mangle_435.ConvBlock
  conv_last3 : __torch__.torch.nn.modules.conv.___torch_mangle_436.Conv2d
  bn_end3 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_437.BatchNorm2d
  l3 : __torch__.torch.nn.modules.conv.___torch_mangle_438.Conv2d
  def forward(self: __torch__.face_alignment.models.fan.FAN,
    input: Tensor) -> Tensor:
    _0 = self.l3
    _1 = self.bn_end3
    _2 = self.conv_last3
    _3 = self.top_m_3
    _4 = self.m3
    _5 = self.al2
    _6 = self.bl2
    _7 = self.l2
    _8 = self.bn_end2
    _9 = self.conv_last2
    _10 = self.top_m_2
    _11 = self.m2
    _12 = self.al1
    _13 = self.bl1
    _14 = self.l1
    _15 = self.bn_end1
    _16 = self.conv_last1
    _17 = self.top_m_1
    _18 = self.m1
    _19 = self.al0
    _20 = self.bl0
    _21 = self.l0
    _22 = self.bn_end0
    _23 = self.conv_last0
    _24 = self.top_m_0
    _25 = self.m0
    _26 = self.conv4
    _27 = self.conv3
    _28 = self.conv2
    _29 = (self.bn1).forward((self.conv1).forward(input, ), )
    input0 = torch.relu_(_29)
    input1 = torch.avg_pool2d((_28).forward(input0, ), [2, 2], [2, 2], [0, 0], False, True, None)
    _30 = (_26).forward((_27).forward(input1, ), )
    _31 = (_24).forward((_25).forward(_30, ), )
    _32 = (_22).forward((_23).forward(_31, ), )
    input2 = torch.relu_(_32)
    _33 = (_21).forward(input2, )
    _34 = (_20).forward(input2, )
    _35 = (_19).forward(_33, )
    input3 = torch.add(torch.add(_30, _34, alpha=1), _35, alpha=1)
    _36 = (_17).forward((_18).forward(input3, ), )
    _37 = (_15).forward((_16).forward(_36, ), )
    input4 = torch.relu_(_37)
    _38 = (_14).forward(input4, )
    _39 = (_13).forward(input4, )
    _40 = (_12).forward(_38, )
    input5 = torch.add(torch.add(input3, _39, alpha=1), _40, alpha=1)
    _41 = (_10).forward((_11).forward(input5, ), )
    input6 = torch.relu_((_8).forward((_9).forward(_41, ), ))
    _42 = (_7).forward(input6, )
    _43 = (_6).forward(input6, )
    _44 = (_5).forward(_42, )
    input7 = torch.add(torch.add(input5, _43, alpha=1), _44, alpha=1)
    _45 = (_3).forward((_4).forward(input7, ), )
    input8 = torch.relu_((_1).forward((_2).forward(_45, ), ))
    return (_0).forward(input8, )
class ConvBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  bn1 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_0.BatchNorm2d
  conv1 : __torch__.torch.nn.modules.conv.___torch_mangle_1.Conv2d
  bn2 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_2.BatchNorm2d
  conv2 : __torch__.torch.nn.modules.conv.___torch_mangle_3.Conv2d
  bn3 : __torch__.torch.nn.modules.batchnorm.___torch_mangle_4.BatchNorm2d
  conv3 : __torch__.torch.nn.modules.conv.___torch_mangle_5.Conv2d
  downsample : __torch__.torch.nn.modules.container.Sequential
  def forward(self: __torch__.face_alignment.models.fan.ConvBlock,
    input: Tensor) -> Tensor:
    _46 = self.downsample
    _47 = self.conv3
    _48 = self.bn3
    _49 = self.conv2
    _50 = self.bn2
    _51 = self.conv1
    input9 = torch.relu_((self.bn1).forward(input, ))
    _52 = (_51).forward(input9, )
    input10 = torch.relu_((_50).forward(_52, ))
    _53 = (_49).forward(input10, )
    input11 = torch.relu_((_48).forward(_53, ))
    _54 = [_52, _53, (_47).forward(input11, )]
    out3 = torch.cat(_54, 1)
    _55 = torch.add_(out3, (_46).forward(input, ), alpha=1)
    return _55
class HourGlass(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  b1_4 : __torch__.face_alignment.models.fan.___torch_mangle_32.ConvBlock
  b2_4 : __torch__.face_alignment.models.fan.___torch_mangle_39.ConvBlock
  b1_3 : __torch__.face_alignment.models.fan.___torch_mangle_46.ConvBlock
  b2_3 : __torch__.face_alignment.models.fan.___torch_mangle_53.ConvBlock
  b1_2 : __torch__.face_alignment.models.fan.___torch_mangle_60.ConvBlock
  b2_2 : __torch__.face_alignment.models.fan.___torch_mangle_67.ConvBlock
  b1_1 : __torch__.face_alignment.models.fan.___torch_mangle_74.ConvBlock
  b2_1 : __torch__.face_alignment.models.fan.___torch_mangle_81.ConvBlock
  b2_plus_1 : __torch__.face_alignment.models.fan.___torch_mangle_88.ConvBlock
  b3_1 : __torch__.face_alignment.models.fan.___torch_mangle_95.ConvBlock
  b3_2 : __torch__.face_alignment.models.fan.___torch_mangle_102.ConvBlock
  b3_3 : __torch__.face_alignment.models.fan.___torch_mangle_109.ConvBlock
  b3_4 : __torch__.face_alignment.models.fan.___torch_mangle_116.ConvBlock
  def forward(self: __torch__.face_alignment.models.fan.HourGlass,
    argument_1: Tensor) -> Tensor:
    _56 = self.b3_4
    _57 = self.b3_3
    _58 = self.b3_2
    _59 = self.b3_1
    _60 = self.b2_plus_1
    _61 = self.b2_1
    _62 = self.b1_1
    _63 = self.b2_2
    _64 = self.b1_2
    _65 = self.b2_3
    _66 = self.b1_3
    _67 = self.b2_4
    _68 = (self.b1_4).forward(argument_1, )
    input = torch.avg_pool2d(argument_1, [2, 2], [2, 2], [0, 0], False, True, None)
    _69 = (_67).forward(input, )
    _70 = (_66).forward(_69, )
    input12 = torch.avg_pool2d(_69, [2, 2], [2, 2], [0, 0], False, True, None)
    _71 = (_65).forward(input12, )
    _72 = (_64).forward(_71, )
    input13 = torch.avg_pool2d(_71, [2, 2], [2, 2], [0, 0], False, True, None)
    _73 = (_63).forward(input13, )
    _74 = (_62).forward(_73, )
    input14 = torch.avg_pool2d(_73, [2, 2], [2, 2], [0, 0], False, True, None)
    _75 = (_60).forward((_61).forward(input14, ), )
    up2 = torch.upsample_nearest2d((_59).forward(_75, ), None, [2., 2.])
    input15 = torch.add(_74, up2, alpha=1)
    up20 = torch.upsample_nearest2d((_58).forward(input15, ), None, [2., 2.])
    input16 = torch.add(_72, up20, alpha=1)
    up21 = torch.upsample_nearest2d((_57).forward(input16, ), None, [2., 2.])
    input17 = torch.add(_70, up21, alpha=1)
    up22 = torch.upsample_nearest2d((_56).forward(input17, ), None, [2., 2.])
    return torch.add(_68, up22, alpha=1)
