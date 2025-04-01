import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 初始化模型、损失函数和优化器
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("Trainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: shape={param.shape}, dtype={param.dtype}")

# 创建输入数据
x = torch.randn(1, 10)
target = torch.tensor([1.0])

# 前向传播
output = model(x)
loss = criterion(output, target)



# 反向传播（保留计算图）
loss.backward(retain_graph=True)

# 检查参数是否在计算图中
print("Checking if parameters are in the computation graph:")
for name, param in model.named_parameters():
    if param.grad_fn is not None:
        print(f"{name} is in the computation graph (grad_fn={param.grad_fn})")
    else:
        print(f"{name} is NOT in the computation graph")

# 释放计算图（如果需要）
loss.backward()  # 正常反向传播，释放计算图