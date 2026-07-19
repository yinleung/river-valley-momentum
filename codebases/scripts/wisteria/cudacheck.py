import torch
print(torch.__version__, torch.version.cuda)
x = torch.zeros(8, device="cuda")
print("cuda init OK:", (x+1).sum().item(), torch.cuda.get_device_name(0))
