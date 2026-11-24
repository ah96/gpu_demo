# gpu_demo

A single script that puts a GPU under training load and reports how it copes. It trains a small convolutional network on the CIFAR-10 training set with PyTorch and prints the throughput in images per second and the GPU memory in use. It is meant for checking that a machine's GPU, driver and PyTorch installation work together, and for seeing how batch size and mixed precision change speed and memory.

## Quick start

Run it on a machine with a CUDA-capable GPU.

```bash
python -m pip install torch torchvision
python gpu_train_stress.py --amp
```

The repo pins no package versions. Install the PyTorch build that matches the CUDA version of the machine.

The first run downloads CIFAR-10 into `./data`, so it needs an internet connection.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--batch-size` | 256 | Batch size. A larger value loads the GPU more. |
| `--epochs` | 5 | Number of passes over the training set. |
| `--lr` | 1e-3 | Learning rate of the AdamW optimizer. |
| `--num-workers` | 4 | Worker processes of the data loader. |
| `--amp` | off | Mixed precision. It takes effect only on a GPU. |
| `--device` | `cuda` | `cuda` or `cpu`. |
| `--print-every` | 50 | Number of steps between two status lines. |

If `--device cuda` is given and no GPU is available, the script runs on the CPU and says so in its first line, `Using device: cpu`.

## What it prints

- At the start: the device, and on a GPU its name, the CUDA version and the PyTorch version.
- At step 1 and then every `--print-every` steps: the loss, the throughput in images per second, and the GPU memory allocated, reserved and at its peak. The value at step 1 covers a single step and includes the start-up of the data loader workers, so it is lower than the later ones.
- After each epoch: the average loss and the time the epoch took.
- At the end, on a GPU: the peak GPU memory allocated over the whole run.

## The model

`SimpleCNN` in `gpu_train_stress.py` has five convolution layers with 64, 128, 256, 256 and 512 channels, each followed by ReLU and batch normalization, three max-pooling steps, and two linear layers with dropout between them. The training images are randomly cropped and flipped, then normalized. The loss is cross-entropy.

## What it does not do

- It does not evaluate on the CIFAR-10 test set, so it reports no accuracy. The training serves as load, and the loss shows only that the training proceeds.
- It sets no random seed, so two runs give different loss values.
- It saves no model and writes no result file. All output goes to the terminal.
- It uses one GPU, the first one PyTorch reports.

## Files

```
gpu_train_stress.py   the model, the training loop and the reporting
```
