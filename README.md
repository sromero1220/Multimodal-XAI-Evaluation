# Multimodal-XAI-Evaluation

# Captum Insights for Visual Question Answering with Added Evaluation of Models

This repository provides an example of using the [Captum Insights API](https://captum.ai/docs/captum_insights) for Visual Question Answering (VQA) along with added functionalities for evaluating different explanation methods.

## Prerequisites

To run this script, you need the following installed on your machine:

### Python Packages
- `torchvision`
- `PIL`
- `matplotlib`

### Repositories
- [pytorch-vqa](https://github.com/Cyanogenoid/pytorch-vqa)
- [pytorch-resnet](https://github.com/Cyanogenoid/pytorch-resnet)

### Pre-trained Model
- A pre-trained pytorch-vqa model, which can be obtained from: [2017-08-04_00.55.19.pth](https://github.com/Cyanogenoid/pytorch-vqa/releases/download/v1.0/2017-08-04_00.55.19.pth)

### Environment Setup
- Create a CUDA environment with `environment.yml` to ensure all dependencies and versions are correct and working.

## Setup

Modify the paths in the script to match your local installation paths:

```python
PYTORCH_VQA_DIR = os.path.realpath("path/to/pytorch-vqa")
PYTORCH_RESNET_DIR = os.path.realpath("path/to/pytorch-resnet")
VQA_MODEL_PATH = "path/to/2017-08-04_00.55.19.pth"
```

Ensure these paths are valid by checking their existence:

```python
assert(os.path.exists(PYTORCH_VQA_DIR))
assert(os.path.exists(PYTORCH_RESNET_DIR))
assert(os.path.exists(VQA_MODEL_PATH))
```

## Running the Script

1. Import necessary libraries and set up paths for the pytorch-vqa and pytorch-resnet directories.

2. Import necessary modules to run the code:
    - `torch`
    - `torchvision`
    - `matplotlib`
    - `PIL`
    - `captum`

3. Load and set up the VQA model, including the vocabulary and answer classes.

4. Modify the VQA model to use pytorch-resnet and set up the device (CPU or CUDA).

5. Define utility functions for image and text inputs:
    - `encode_question`
    - `baseline_image`
    - `baseline_text`
    - `input_text_transform`

6. Use the Captum Insights API for visualization:
    - Define features for images and text
    - Create an `AttributionVisualizer` object
    - Visualize the model's outputs on local host and run a XAI model

7. Add perturbation functions for sensitivity analysis:
    - `perturb_image`
    - `perturb_text`

8. Evaluate the models using various metrics:
    - AOPC Comprehensiveness
    - AOPC Sufficiency
    - Plausibility

9. Run the sensitivity test and display results.

10. Display a screenshot if using the notebook non-interactively.

## Example Usage

```bash
python Multimodal_VQA_Captum_Insights.py
```

## Additional Information

For more details on the Captum Insights API and its functionalities, refer to the [Captum documentation](https://captum.ai/docs).

For details on the VQA model interpretation tutorial, refer to [this tutorial](https://captum.ai/tutorials/Multimodal_VQA_Interpret).

