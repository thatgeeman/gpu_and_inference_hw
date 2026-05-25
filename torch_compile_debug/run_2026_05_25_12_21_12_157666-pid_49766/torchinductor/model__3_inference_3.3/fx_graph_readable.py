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
        add_7: "f32[67108864]" = torch.ops.aten.add.Tensor(mul_7, arg0_1);  mul_7 = arg0_1 = None
        return (add_7,)
        