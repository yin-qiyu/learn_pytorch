��
l��F� j�P.�M�.�}q (X   protocol_versionqM�X   little_endianq�X
   type_sizesq}q(X   shortqKX   intqKX   longqKuu.�(X   moduleq cmodel_save
Tudui
qX>   /Users/yinqiyu/PycharmProjects/learn_pytorch/src/model_save.pyqX�   class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x
qtqQ)�q}q(X   trainingq�X   _parametersqccollections
OrderedDict
q	)Rq
X   _buffersqh	)RqX   _backward_hooksqh	)RqX   _forward_hooksqh	)RqX   _forward_pre_hooksqh	)RqX   _state_dict_hooksqh	)RqX   _load_state_dict_pre_hooksqh	)RqX   _modulesqh	)RqX   conv1q(h ctorch.nn.modules.conv
Conv2d
qX^   /Users/yinqiyu/opt/anaconda3/envs/pytorch/lib/python3.6/site-packages/torch/nn/modules/conv.pyqX�  class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters, of size:
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    .. note::

         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.

    .. note::

        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also termed in
        literature as depthwise convolution.

        In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`,
        a depthwise convolution with a depthwise multiplier `K`, can be constructed by arguments
        :math:`(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})`.

    .. include:: cudnn_deterministic.rst

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{1}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples::

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)
qtqQ)�q}q(h�hh	)Rq (X   weightq!ctorch._utils
_rebuild_parameter
q"ctorch._utils
_rebuild_tensor_v2
q#((X   storageq$ctorch
FloatStorage
q%X   105553175053920q&X   cpuq'M�Ntq(QK (K@KKKtq)(KK	KKtq*�h	)Rq+tq,Rq-�h	)Rq.�q/Rq0X   biasq1h"h#((h$h%X   105553175054016q2h'K@Ntq3QK K@�q4K�q5�h	)Rq6tq7Rq8�h	)Rq9�q:Rq;uhh	)Rq<hh	)Rq=hh	)Rq>hh	)Rq?hh	)Rq@hh	)RqAhh	)RqBX   in_channelsqCKX   out_channelsqDK@X   kernel_sizeqEKK�qFX   strideqGKK�qHX   paddingqIK K �qJX   dilationqKKK�qLX
   transposedqM�X   output_paddingqNK K �qOX   groupsqPKX   padding_modeqQX   zerosqRubsub.�]q (X   105553175053920qX   105553175054016qe.�      Gp � �<�U�U���50>4c=/>(�3=[�����"/>b�	�N��=�o(��E >�����+=N�2�Ĥ��=j�=)Sݽ��>&X�k�%���<F��=����tp�pP <L,L=�b彏�׽ g�;��>��Ľ�3#= �~�D�0=)� >n��=�l
>����V�=5�>~t�=x|�<�j�=���=�T>%t��6!�=���o1D��ݽ�����Ǵ=���H;�<��	X>\�0���ս�q̽ �*��ػ8�b�hs0=�i->l�O=ӭ&�P&��~��X.��E7�'��t�`� �˻���>���=&&�\n=Zڴ=��jg�=���Lh��J8�9C>�W;<9�0>���Xs��`ǽ���(ռ?[3> �B��Г=��=X&5�n1�=>��=�����ʧ=�r�����=˶<��H�<��}�2>�ۦ=Pd�3Tý���&��=���<��"�)�>�f*�����>J�=L�D�i!�F?�=JP�=��U=��'>�SB=���.{�=�#>����ƽH�2����=�J���ek=Qs>�bཏm7>*�3�X�@=hM=�:�=�-�*�>�f�:�\Rp=L��@���W=}�Z��=��K=Z�=pl�����<`<F�X�<��;��> ��:*ٹ�/C�lL5���8�<BI=Q���z>�S%=]D>�u�=������� ���"��@*B=��B�l*�+a>h��<v8��Gs���<��½�J=�X�;��>����<��<�&�w�ܽ� '�:��=��-�@���/�^4�=�E�;ص"=(���Q�
> 3:�ɹ�����z6>(�c=���=D�u="��=Y�ʽ��(���I&>�B�6ϑ=ZT��#���>��>�>�p���f>�R�<?��g<�L$���)��u=?*>��=�h�=���<j9"�]<�z�����o;�@��̜=Z��=��>	N><�@�f�Ž.�����y=y6��s���4=m�lR��`5̼���=���b�,��� �n��=���L�g�?@���D>Lgʽ�?>��Y�=,�%�f==�e��*���#> |�� �]�*i�

�=(O�<�p�=�+:����=�
���"�<�z	>K4�� a=ű>���<���=��(QC=)�$�qV뽰A�p�E=3'�'�m|���3��j�=@|�<�A=��=6����=�m�������
�=�" =4�#��I���wý������^��=([����)>
������NO��a�<D =��-���$��=�9q= �q:P\u<$�,�;����9�*� ���=Pw����= �\�_=S�A>E����ɽ�n!>l�4�p=����=�!Z=���i�ݽ�6����=@?�<������=���<~9�=R��:f��*����ب��<7���n<�|�=�X>�%佒v��,�ؽ��򻐧F=�=m�<ĈT���>�s6>[ƽ��]���V��H�=�N	>,���ۄ�1tƽ�h����$���o�� >`�g�&Z�=X��<ށ�=��{�"/�r�@��*������*>�E��5*>�e��;�=2a�s=���Q�7>�:
>���6>�>nN��@�w��׽Qx >.�Ƚ�������l=$Ei=��=��9>X��<�-�j�.��=R|�=��� ��<x�=�p==� ���|/w=!�$>$�*�H}�%~3>��<>`L�;t>���I>R�ƽ��>`�����@�~<H�<=ɷ:�'\	>��,>�;�<�>�뽉��$ �G�>ۯ�`�j<��;�x>Y�W�=���=6�4�(#�<$>Ӷ��0�O>�^2=؇����i=��'=����f���I0>�> ���8>�V>�L�=�.���轿����)����=ѕ�6��=�0�<(?>ß̽��>��F�f�=Ns�=�0=�O=����b�x����>�{ؼ��>V��=��>��t=>c�5��	�˺->��=�������=��> @�;&��=
�=�k= jw�X!�����z�=�'�<���=-0>�>W6B>�W>���U�$>,NJ=
��=.�=[�8�x6ݼ���<z蒽��e��=A�����)�?>!y#>x�.�ig>��)>r��=2>''>X^��>U`&>�v��8l=�"@��">�9>���=p&=��8>�p@�P�����=����(�������<�?�&o�=Tf����=y�&��S>�->�%;�@O;���0��������,�9F=�y�=��-=rE�>ם=�<�=\j;���*�X>R׼=���x�3�d�K�ɴ>y=�?���4��޽�>�?�T����+*>>D�=R�=S���D�/�A�w�+>�?�&A>��佳�0�L�~=PC��ƹ�="���A��9W���>�8q��֯=��:��+���
�����=|�s=�>j��=��7>�̝� '�����=������=�
>>>w@����`�0=_�
>�>�K޽:у=}�'�@ڽ�+����7>�X��-�آa=蔽���K�=�t�=�t���Z<�d0��>�=7> �һ��ӽ�L�pDA<���=�\�l����.���H=v��R���8�� ����+=)��������N|= ��0�=�=�����*����*">�s<>�k	>Z��=&��⒕=��>��A���m=ה5>&�=T�4�q쮽���=�C>8빽G�=�����֜�Q&�
��)g�;j>`1
= ��;`&��);�*�=CYA>[ >�e=��,���]=��=�߈��韽��@�F��=<�~�B~����<��=�g4>lr��t�M�؋a��=8[��\�=�W�=������ =�^��p\�Y����=�_)>��m=��s�D�j�3Y">hs��NK�=�$>6��=���<�}S�������;�G>S�Ƚ�T>L=za�=v:�=�>�>�,R�t��[C���>��;��3�X� ���=�u���(=��"�"�=��Z���==�F�=ƫ��w*>�Y��%,��)T�q˶���X= ��>��=M�=/g@�������<�|=:�=~Z��k8��,&�=�[
>8�a�������N���-�=�\�����=���g��8��8O6=�O:�Y��j��=�/½>��=x��$¼��G<@a�����=]�9�z,�=>��=:���E�9����Y2��u�=@u�;�d�Ȅ��ӟ����=��7�Z��=Y�B>�`�=F��=JF�=��۽<�g=�">4����#>��6> �9��7>/���&��b=������>��=R�����=�z>(�$���5����d�=t����@��ƀ�@�_<q=@��`u��."���
>���=�#+�1:#>��t=�$��.V��p�=�L��=���s���c�=�N�=�@��;� Xb;�2��8�=�M(=������;>r	��5 3>P�=�Ok��k�=:�=��r����n!"���=����ٌ���� >� �<삼O�4>�Ѹ�����vA�=u;����=C�0>c	>(�2�V��=𗯽���0 9��Ҳ=!�@����= �h����<
�2��=ڃ�=$q��߿����<�J����� 8�;�56>�p�4u{��8�� =Å1>`.�;�RD�n���һ�ހ�=p�>�����B�=��<�P5��'��&��=p6"�C��&����ü�>0>=�>!�>>��=�	>�=?�:�=�Cg=�Oҽ� ���s������e=���K�= �9<��ν>܄k�ǹ>L���"��5>-=�$��A���`���p�=���=w�:>Xo{�v��=P�,<�%z=�G�=o�< �*�-�_� �H�ܽ�K�� 	`<ln��<�m�C><yF=v1������D>,6�H��<{�Ӭ彎�� +���.>ҍ�=[�	��->� �=ˤ:>�o�= DϹfo�=o�>���=�0">�Y��;0��"�=+�ƽ>9�=P�=;�+>���=d5� Ce�E�->*��=�ڍ<��=%/�E��GP*>��=5#��={T6>�߼=�v��+�>�85>�Q��En8> aỴ�`=d�r�ֽ�Ӏ�q�<>�>�>]� >�<�_9��~{;��#>Z��= �<�� >�1ս�=��L='�)��V�<	|�N'�=�
��Q��v@�=�X� �
����= ����<]�-> 
���H�˅����=	!>k�=>wƽx�<8�ܼ@�;���=��p*D= u<���� ��;W�/�z͠=.)�=���=�T⽊��=r�=<�>=� ->�>�����=-~ؽ�7���=� �'�>V�����><�����t'>�"
=rω=�h�=���=��	>��>����=\iG���k=��=��?>Q>â='&	>��<(NW=������H�M�>�]C>�򏽨��<ڱ� �㼀n��!��MS�Y�>�fz�ܬ4�꬀���<~#���=<�<ݘ-��Uv= ��<�	B��,;=e�)��[ =�h<�&=������=��D>Ol>e���6߽v?�=dZ�[L:>`퓻�=�I>�*���">ҙս��m=��^/�=HV9=ab(>ﴌ���=PK�<����"���T�=6�����`�� N���?Y������
�=�&�����. >�O	>�n������S�=������=���G�>�+
��=>��;>L�.�="&>�K;�11>*������D����<��'�ҭ�d3���K��Rc���,>̠m�,ʽ��0>r�=��#>�$�pa��h2<�G�Pt�<pg]����M�$>�$�L��9��=����R��=�p��Ȑ�������=,�Z�ZΑ= �%�)!��ο=^��=�"j=��<T'��۬��I4�@��w&潶D�=�.>P��a1�3`�4\p=֦�=�*�=ؽ�<��=p�M=�����.=�6Ǽ��.�`JC</�����E콇���(=���<��= @<�A;�*"�=Vl���;�X�=];����=���=Vڈ=�t�Ε�=� ��꿽8�����	>��=p�J��Z�=h�<U<��>p:(�z`ѽ�?��^R�{>>D?D=�=d��� ֻ��@>����6�=�I�<ב>(<�	�>��K����=׍���C�=�=�� �1�6 �=lbn��zC��>�=3�>k>Qͼ���-��$���?���=#R=���!=nw�@>/C>�r˼�-�|�L���������v�=l1���5�f���P`6<H���h�=hi`=���;XD�����0�r<����E?>�����ud�]Q��  �����=���0T��D ���*=�ȼ�b:�=`�(����=�=z����= @�f��= �&�2��=���x��<����0�
� ��<X�J��㎽����>����<?O>�3>���a2��r�Ͻ�>$��Ɛ��B>�#�����;�h�=I�>��<;V0�Y퐽�0>L�.�۱>��:=�W��<�M=�l'>0�ϼC<����B> �I;M�Tz��%=贉��.=&
�&>۷>(J�d�;��cJ��'̽�
5;������˦�H�	=��<0�6��H����=� X����<R,�=,km���"�Pp<���;�R���W3=���ϳ���>�&���!��5"����7:5>�lg=��>��: �����#>̀=�
> 4��w�=H9�HӼ�0:�q>�4>@|��<r��=9>�H�>=��+�j �='�P��<$�=���=�>��<[�B> �ʼV��=�G6��L>��=0,�����;(��h���j7��t>ǁ>��;5��5>�q���f=��˽��<9�A�����@�h�̼�K��M���B�ּ�=������;ؙ�<�k�=U��U�=�k�=�Z�l	C�(+Z=�lv��c���&>�BS�O=�����D�=�e�=�~���~�=Bh�=����D���l>θ��B>Z��=ґ�=vV����>��� ݦ<^)���j=�,B��Β����=_�>e�==�>P�̼b��=k콐��P/=?sɽ��2=�a�� M�<��=�j��=�s>
��=Ɠ����>h6>��>U�>��=��l��� ���^�?/��L���ƽ��$��)>g����-�ST�8�5��A>>ʹ'��������=���m�(� �-<f	�=��d=z�=�q�<J��k�>�dͽ/��`�߼�0��H�C�?>��|u �����轺�C��D>�+@�Ů5>�N��z��=\x1=g7���v#=q\#>;i2>|�=�ʉ�0b��Q@��=^y�=�?=�ڵ=tRg��a>��0�`DI<�O�<x������=�B>8�=�/6<��.>��;�`���B�d������^�����=��޻�B��䏽�j=t#�I�ϽJ-�=�h!;�-=`?0���t��_=���?�=�� ���<Wi*>0��<�/�<X��<�/>�^ּ�p��<jy��L(>�p>�=���0A���Խ�/9���=@z�<�'C��@���0��f=[�4���D>f7�=@�`� K+�*��=���=l�{=�N���@>4l=P\ȼ�T�=�g���W=1d������	>��)>l�{��aX=�j,>R4�=@g}���k=���=@_��d����H<>h���A>�7�I,ƽ� ���'��@g�;W 5��;u�x�ͽ��3 >���<M#>���=�mA��x�@       �B�d����D��)�;Ʈ��h��l� �|�@�λ�h>`�<t�=��=|8���$���aٽ'��z�0�+4>B䂽  �;��=�<��{�>_����U=�47�U�!p2�Hd)=��4��;= �H<�v@�gE�]� �|�6�B˟=��
>"��=�x���L=Urɽ�[���潒��=���=�0�b�.I��,>���/m@��3>c�>�=�=��6= T�`�s<>��=�T�=�H"�"<�