# Fine-Tuning Diffusion Models with LoRA: Golden Retrievers

## Overview

This project implements **Low-Rank Adaptation (LoRA)**, an efficient fine-tuning technique, to adapt a pre-trained Stable Diffusion model for generating high-quality images of Golden Retrievers. The project explores the effectiveness of LoRA with different rank configurations and demonstrates its capability to achieve competitive results with minimal computational overhead.

## Project Objectives

1. **Understand LoRA Principles**: Master the theoretical foundations and practical implementation of Low-Rank Adaptation for efficient neural network fine-tuning.

2. **Dataset Preparation**: Curate and prepare a high-quality dataset of Golden Retriever images for controlled fine-tuning experiments.

3. **Multi-Rank Experimentation**: Implement LoRA fine-tuning with different rank parameters (r=8 and r=16) to evaluate the trade-offs between model capacity and training efficiency.

4. **Output Generation & Comparison**: Generate and compare images from the base model and LoRA-enhanced variants to demonstrate the quality improvements.

5. **Performance Analysis**: Measure and analyze key metrics including trainable parameters, inference time, and VRAM consumption.

## Key Features

- ✅ **Efficient Fine-Tuning**: LoRA reduces trainable parameters to ~0.4% while maintaining quality
- ✅ **Multi-Rank Configurations**: Experiments with ranks 8 and 16 to explore quality-efficiency trade-offs
- ✅ **GPU-Optimized Training**: Implements gradient accumulation and mixed precision for efficient VRAM usage
- ✅ **Comprehensive Metrics**: Tracks training progress, inference performance, and memory consumption
- ✅ **Special Token Integration**: Uses custom token `<sksdog>` for precise concept grounding

## Project Structure

```
.
├── LORA.ipynb                          # Main notebook with all experiments
├── README.md                           # Project overview (this file)
├── REPORT.md                           # Detailed technical analysis
├── Golden/                             # Input dataset (Golden Retriever images)
├── Golden_Output/                      # Processed dataset
│   ├── images/                         # Resized images (512×512) and captions
│   └── metadata/                       # Dataset metadata and statistics
├── models/                             # Trained LoRA weights
│   ├── lora_sksdog_r8/                 # LoRA with rank 8
│   └── lora_sksdog_r16/                # LoRA with rank 16
└── outputs/                            # Generated images from inference
```

## Dataset

**Golden Retrievers Dataset**
- **Source**: Curated collection of Golden Retriever images
- **Size**: 100+ high-quality images
- **Processing**:
  - Resized to 512×512 pixels (standard for Stable Diffusion)
  - Converted to RGB format
  - Paired with descriptive captions
  - JPEG compression at 95% quality

**Caption Format**: `"a photo of a <sksdog> dog"`

The special token `<sksdog>` enables the model to learn breed-specific visual characteristics while remaining generalizable. Honestly, I could have used `<goldenretriever>`, but I when I started I was working with Husky images and then with Dashounds. So when I started working with Golden Retrievers I didn't know if they were going to be the last breed I would work with. Hence, I used `<sksdog>` as a more generic token for "some breed of dog".

## Implementation Details

### Architecture

- **Base Model**: Stable Diffusion v1.5
- **Adapter Method**: LoRA (Low-Rank Adaptation)
- **LoRA Config**:
  - Target modules: `to_q`, `to_k`, `to_v`, `to_out.0` (attention layers)
  - Dropout: 0.05
  - Bias: None

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Learning Rate | 1e-4 |
| Scheduler | Cosine with warmup |
| Training Steps | 2000 |
| Image Resolution | 384×384 |

**Note**: Image size reduced to 384×384 for efficient 4GB GPU training while maintaining quality. Because my GPU has only 4GB of VRAM, I had to reduce the image size from the standard 512×512 to 384×384. This allowed me to train the model effectively without running out of memory. (It still took 10 hours to train) 

### Training Optimization Techniques

1. **VAE & Text Encoder Frozen**: Only LoRA layers in the UNet are trainable
2. **Gradient Checkpointing**: Reduces peak memory usage
3. **CPU Offloading**: Non-essential computations moved to CPU
4. **Mixed Precision**: float32 for stability with modest VRAM gains
5. **Gradient Accumulation**: Effective batch size of 4 with physical batch of 1

## Usage

### 1. Data Preparation

```python
prepare_dataset(
    golden_folder="./Golden",
    output_folder="./Golden_Output"
)
```

This will:
- Resize and normalize all images
- Create caption text files for each image
- Generate metadata JSON with dataset statistics

### 2. Training LoRA Models

```python
train_lora(
    images_dir="./Golden_Output/images",
    output_dir="./models/lora_sksdog_r8",
    rank=8,           # or 16 for higher capacity
    steps=2000,
    lr=1e-4
)
```

Training takes approximately 60-90 minutes per rank on a 4GB GPU.

### 3. Inference

```python
from diffusers import StableDiffusionPipeline
from peft import PeftModel

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA weights
pipe.unet = PeftModel.from_pretrained(
    pipe.unet,
    "models/lora_sksdog_r8"
)

# Generate image
image = pipe(
    "a photo of a <sksdog> dog in a park",
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]
```

## Results

### Parameter Efficiency

| Model | Trainable Params | Total Params | % Trainable |
|-------|------------------|--------------|-------------|
| Base Model | 0 | 860M | 0% |
| LoRA r=8 | 3.5M | 860M | 0.41% |
| LoRA r=16 | 7.0M | 860M | 0.81% |

### Inference Performance

Memory and speed metrics measured during image generation (30 inference steps, guidance_scale=7.5):

- **Peak VRAM**: ~4.2 GB
- **Inference Time per Image**: ~12-15 seconds
- **Batch Generation**: Supported for higher throughput

### Generated Outputs

The two LoRA models showed distinctly different results:

**LoRA r=8**: Generates generic dogs
- Produces recognizable dog images
- Learning limited to general canine features
- Does NOT successfully capture Golden Retriever-specific characteristics
- Insufficient for breed-specific generation

**LoRA r=16**: Successfully generates Golden Retrievers
- Accurate breed-specific visual characteristics
- Improved color and texture accuracy (golden coat, characteristic facial features)
- Consistent performance across varied prompts and environments
- Higher rank provides necessary capacity for precise breed learning

## Key Findings

1. **LoRA Efficiency**: Training only 0.4-0.8% of parameters achieves breed-specific results on consumer GPUs
2. **Critical Rank Threshold**: r=8 generates generic dogs but fails to capture breed specificity. r=16 successfully generates Golden Retrievers with strong visual accuracy. The 2x parameter cost is necessary to achieve breed-level precision.
3. **Rank Sufficiency**: r=16 appears to be the minimum viable rank for breed-specific image generation with this model
4. **Training Stability**: Special token approach (`<sksdog>`) is effective for concept learning, though higher rank required for precise breed characteristics
5. **Memory Efficiency**: LoRA enables fine-tuning on consumer GPUs (4GB) that would otherwise require 24GB+ for full training
6. **Inference Overhead**: Negligible performance impact; LoRA adds <5% latency

## Dependencies

```
torch>=2.0
diffusers>=0.21
transformers>=4.30
peft>=0.4
accelerate>=0.20
pillow>=9.0
```

Install via:
```bash
pip install torch diffusers transformers peft accelerate pillow
```



## References

1. **LoRA Paper**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
2. **Stable Diffusion**: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) - Rombach et al., 2022
3. **Hugging Face Diffusers**: https://huggingface.co/docs/diffusers/
4. **PEFT Library**: https://huggingface.co/docs/peft/

## License

This project is provided as-is for educational and research purposes.

## Author

Francisco González García

---

**Last Updated**: January 2026
