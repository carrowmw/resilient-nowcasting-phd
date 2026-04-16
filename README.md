# ML Agent Workspace

## Build Instructions

To build the agent workspace container on your local machine, run the following command in the terminal:

```Bash
docker compose build agent-workspace
```

This will build the container that you will use for development on your Mac.

```Bash
docker compose build cpu-runner
```

When you are ready to execute your training script on the GPU server, you will need to build the CUDA image. Run the following command on the server:

```Bash
docker compose build gpu-runner
```

## Execution Instructions

### On Mac (Development):

You will boot the interactive terminal exactly as you have been doing to let the agent code:

```Bash
docker compose run --rm agent-workspace bash
```

### On the GPU Server (Execution):

Once you have pushed your main branch to GitHub and pulled it down to the server, you will run the GPU container to execute the W&B training script:

```Bash
# This builds the CUDA image and runs your training script
docker compose run --rm gpu-runner python src/train.py --epochs 50 --batch_size 128
```
