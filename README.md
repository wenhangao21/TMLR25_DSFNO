# [TMLR 2025] Dynamic Schwartz-Fourier Neural Operator for Enhanced Expressive Power

This repository contains the code implementation for the paper titled "Dynamic Schwartz-Fourier Neural Operator for Enhanced Expressive Power"

Wenhan Gao, Jian Luo, Ruichen Xu, Yi Liu

Transactions on Machine Learning Research (TMLR), 2025

Full paper on [OpenReview](https://openreview.net/forum?id=B0E2yjrNb8).

**Bibtex**:
```bibtex
@article{
gao2025dynamic,
title={Dynamic Schwartz-Fourier Neural Operator for Enhanced Expressive Power},
author={Wenhan Gao and Jian Luo and Ruichen Xu and Yi Liu},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=B0E2yjrNb8},
note={}
}
```

## Introduction 
> This paper investigates the expressiveness of Fourier Neural Operators (FNOs). **In particular, convolution kernels can be seen as a special case of integral kernels.** FNOs approximate integral operators using convolutions in the Fourier domain to leverage the efficiency of fast Fourier transforms (FFTs). However, this approximation limits their expressive power. **To address this, we propose a method for performing general integral kernel operations directly in the Fourier domain and introduce a more efficient implementation of this approach.** We also provide theoretical justification showing that these operations are equivalent to kernel integrals in the physical (spatial) domain.

## Datasets

We use the same Navier-Stokes and Darcy flow dataset as provided by the FNO authors, which is available [here](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing), courtesy of the original authors. The shallow water equation is from the [PDEBench](https://github.com/pdebench/PDEBench).

## How to Run

Following the style of the [FNO repository](https://github.com/lilux618/fourier_neural_operator), we provide our implementation for the Navier–Stokes example as a simple, stand-alone script. The script is self-contained and directly runnable.

```
python3.12 DSFNO.py 
```

Overall, this implementation is a simple modification of the original FNO architecture. We replace the spectral convolution layers with the spectral integration layers given below. In practice, we found that replacing only the first or the last spectral layer typically yields the best performance and leads to more stable training.

```python
class SpectralInt2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralInt2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.spectral_weight = Spectral_weights(in_channels=in_channels, out_channels=out_channels, modes1=modes1, modes2=modes2)
        self.spectral_weight1 = Spectral_weights(in_channels=in_channels, out_channels=out_channels, modes1=modes1, modes2=modes2)
        self.spatial_mixing_1 = nn.Parameter(1/modes1 * torch.rand(out_channels, modes1*2-1, modes1*2-1, dtype=torch.cfloat))
        self.spatial_mixing_2 = nn.Parameter(1/modes2 * torch.rand(out_channels, modes2, modes2, dtype=torch.cfloat))
        
        self.get_weight()

    def get_weight(self):
        self.spectral_weight.get_weight()
        self.weights = self.spectral_weight.weights
        self.spectral_weight1.get_weight()
        self.weights1 = self.spectral_weight1.weights
        
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)
        
    def mul2d_sp_1(self, input):
        weights = self.spatial_mixing_1
        return torch.einsum("bcix,cio->bcox", input, weights)

    def mul2d_sp_2(self, input):
        weights = self.spatial_mixing_2
        return torch.einsum("bcyi,cio->bcyo", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # get the index of the zero frequency and construct weight
        freq0_y = (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        self.get_weight()
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
        x_ft = x_ft[..., (freq0_y - self.modes1 + 1):(freq0_y + self.modes1), :self.modes2]
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)                              
                             
        x_ft = self.compl_mul2d(x_ft, self.weights)
        x_ft1 = self.mul2d_sp_1(x_ft)
        x_ft1 = self.mul2d_sp_2(x_ft1)
        x_ft1 = self.compl_mul2d(x_ft1, self.weights1)
        x_ft = x_ft*x_ft1 + x_ft 
        out_ft[..., (freq0_y - self.modes1 + 1):(freq0_y + self.modes1), :self.modes2] = x_ft
        # Return to physical space
        x = torch.fft.irfft2(torch.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1)))
        return x
```

