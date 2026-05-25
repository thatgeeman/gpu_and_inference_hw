class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[67108864]"):
         # File: /home/gg/nebius/gpu_and_inference_hw/hw1/hw1_task_impl.py:53 in fn, code: acc = acc * x + x
        mul: "f32[67108864]" = torch.ops.aten.mul.Tensor(arg0_1, arg0_1)
        add: "f32[67108864]" = torch.ops.aten.add.Tensor(mul, arg0_1);  mul = None
        mul_1: "f32[67108864]" = torch.ops.aten.mul.Tensor(add, arg0_1);  add = None
        add_1: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_1, arg0_1);  mul_1 = None
        mul_2: "f32[67108864]" = torch.ops.aten.mul.Tensor(add_1, arg0_1);  add_1 = None
        add_2: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_2, arg0_1);  mul_2 = None
        mul_3: "f32[67108864]" = torch.ops.aten.mul.Tensor(add_2, arg0_1);  add_2 = None
        add_3: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_3, arg0_1);  mul_3 = None
        mul_4: "f32[67108864]" = torch.ops.aten.mul.Tensor(add_3, arg0_1);  add_3 = None
        add_4: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_4, arg0_1);  mul_4 = None
        mul_5: "f32[67108864]" = torch.ops.aten.mul.Tensor(add_4, arg0_1);  add_4 = None
        add_5: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_5, arg0_1);  mul_5 = None
        mul_6: "f32[67108864]" = torch.ops.aten.mul.Tensor(add_5, arg0_1);  add_5 = None
        add_6: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_6, arg0_1);  mul_6 = None
        mul_7: "f32[67108864]" = torch.ops.aten.mul.Tensor(add_6, arg0_1);  add_6 = None
        add_7: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_7, arg0_1);  mul_7 = None
        mul_8: "f32[67108864]" = torch.ops.aten.mul.Tensor(add_7, arg0_1);  add_7 = None
        add_8: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_8, arg0_1);  mul_8 = None
        mul_9: "f32[67108864]" = torch.ops.aten.mul.Tensor(add_8, arg0_1);  add_8 = None
        add_9: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_9, arg0_1);  mul_9 = None
        mul_10: "f32[67108864]" = torch.ops.aten.mul.Tensor(add_9, arg0_1);  add_9 = None
        add_10: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_10, arg0_1);  mul_10 = None
        mul_11: "f32[67108864]" = torch.ops.aten.mul.Tensor(add_10, arg0_1);  add_10 = None
        add_11: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_11, arg0_1);  mul_11 = None
        mul_12: "f32[67108864]" = torch.ops.aten.mul.Tensor(add_11, arg0_1);  add_11 = None
        add_12: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_12, arg0_1);  mul_12 = None
        mul_13: "f32[67108864]" = torch.ops.aten.mul.Tensor(add_12, arg0_1);  add_12 = None
        add_13: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_13, arg0_1);  mul_13 = None
        mul_14: "f32[67108864]" = torch.ops.aten.mul.Tensor(add_13, arg0_1);  add_13 = None
        add_14: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_14, arg0_1);  mul_14 = None
        mul_15: "f32[67108864]" = torch.ops.aten.mul.Tensor(add_14, arg0_1);  add_14 = None
        add_15: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_15, arg0_1);  mul_15 = arg0_1 = None
        return (add_15,)
        