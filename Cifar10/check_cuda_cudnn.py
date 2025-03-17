import torch

def check_cuda_cudnn():
    print("CUDA available:", torch.cuda.is_available())
    print("cuDNN version:", torch.backends.cudnn.version())
    print("PyTorch version:", torch.__version__)  # 查看当前的 PyTorch 版本

if __name__ == '__main__':
    check_cuda_cudnn()
