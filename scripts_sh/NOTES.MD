## set up

### Allocate Ressources



### Modules
```bash
module load release/23.04  GCC/12.2.0 Python/3.10.8 OpenMPI/4.1.4 CUDA/11.8.0
```

### Virtual Enviroment

one enviroment `.venv` for both repositories `structure-tensor` and `3dseg`

#### Packages

most importnant packages

- `CuPy` (with `CUDA v11.x` support; for other CUDA version or AMD GPU support see [documentation](https://docs.cupy.dev/en/stable/install.html))

```bash
python -m pip install cupy-cuda11x
```
check installation

```bash
python -c "import cupy as cp; x = cp.array([1, 2, 3]);print(x.device);print(cp.cuda.runtime.getDeviceCount());cp.cuda.Device(0).use()"
```

- `PyTorch==2.1.2` mit `CUDA==11.8.0`
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- some issues with CUDA 12.0 
- [x] clarify the CUDA version  

check installtion
```bash
python -c "import torch; print(torch.__version__); print(torch.__path__);print(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))"
```

## MCRPy 

