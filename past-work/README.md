## Setup
Easiest setup is to use the same instance type + AMI as me then run the config script I had an LLM generate to install all dependencies for CUDA and OpenCV related things. 

Instance Type: g4dn.2xlarge
Min Storage: 75GB
AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20250523

Running script:
```bash
chmod +x ./scripts/setup_instance.sh
./scripts/setup_instance.sh
```

If you want to set it up on tetracosa you need the prof to setup NVML (or if you are the prof then go for it). With permissions it should be easy.

## Running 
```bash
makedir build
cd build
cmake ..
make run_video_experiments
./run_video_experiments
```