class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[67108864]"):
         # File: /home/gg/nebius/gpu_and_inference_hw/hw1/hw1_task_impl.py:53 in fn, code: acc = acc * x + x
        mul: "f32[67108864]" = torch.ops.aten.mul.Tensor(arg0_1, arg0_1)
        add: "f32[67108864]" = torch.ops.aten.add.Tensor(mul, arg0_1);  mul = arg0_1 = None
        return (add,)
        