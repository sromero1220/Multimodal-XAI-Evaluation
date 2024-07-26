# %% [markdown]
# # Captum Insights for Visual Question Answering with Added Evaluation of Models

# %% [markdown]
# This notebook provides a simple example for the [Captum Insights API](https://captum.ai/docs/captum_insights), which is an easy to use API built on top of Captum that provides a visualization widget.
# 
# 
# As with the referenced tutorial, you will need the following installed on your machine:
# 
# - Python Packages: torchvision, PIL, and matplotlib
# - pytorch-vqa: https://github.com/Cyanogenoid/pytorch-vqa
# - pytorch-resnet: https://github.com/Cyanogenoid/pytorch-resnet
# - A pretrained pytorch-vqa model, which can be obtained from: https://github.com/Cyanogenoid/pytorch-vqa/releases/download/v1.0/2017-08-04_00.55.19.pth
# - Create a CUDA environment with environment.yml do all dependencies and versions are correct and working
# 
# Please modify the below section for your specific installation paths:

# %%
import sys, os

# Replace the placeholder strings with the associated 
# path for the root of pytorch-vqa and pytorch-resnet respectively
PYTORCH_VQA_DIR = os.path.realpath("C:\\Users\\saroa\\OneDrive\\Documentos\\XAI\\pytorch-vqa")
PYTORCH_RESNET_DIR = os.path.realpath("C:\\Users\\saroa\\OneDrive\\Documentos\\XAI\\pytorch-resnet")

# Please modify this path to where it is located on your machine
# you can download this model from: 
# https://github.com/Cyanogenoid/pytorch-vqa/releases/download/v1.0/2017-08-04_00.55.19.pth
VQA_MODEL_PATH = "models/2017-08-04_00.55.19.pth"

assert(os.path.exists(PYTORCH_VQA_DIR))
assert(os.path.exists(PYTORCH_RESNET_DIR))
assert(os.path.exists(VQA_MODEL_PATH))

sys.path.append(PYTORCH_VQA_DIR)
sys.path.append(PYTORCH_RESNET_DIR)

# %% [markdown]
# Now, we will import the necessary modules to run the code. Please make sure you have the [prerequisites to run captum](https://captum.ai/docs/getting_started), along with the pre-requisites to run this tutorial (as described in the first section).

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

try:
    import resnet  # from pytorch-resnet
except:
    print("please provide a valid path to pytorch-resnet")

try:
    from model import Net, apply_attention, tile_2d_over_nd  # from pytorch-vqa
    from utils import get_transform  # from pytorch-vqa
except:
    print("please provide a valid path to pytorch-vqa")
    
from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature, TextFeature
from captum.attr import TokenReferenceBase, configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

# %%
run_on='cuda'  # change to 'cuda' if a GPU is available
if run_on == 'cuda':
    # Let's set the device we will use for model inference
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# %% [markdown]
# # VQA Model Setup
# 
# Let's load the VQA model (again, please refer to the [model interpretation tutorial on VQA](https://captum.ai/tutorials/Multimodal_VQA_Interpret) if you want details)

# %%
saved_state = torch.load(VQA_MODEL_PATH, map_location=device)

# reading vocabulary from saved model
vocab = saved_state["vocab"]

# reading word tokens from saved model
token_to_index = vocab["question"]

# reading answers from saved model
answer_to_index = vocab["answer"]

num_tokens = len(token_to_index) + 1

# reading answer classes from the vocabulary
answer_words = ["unk"] * len(answer_to_index)
for w, idx in answer_to_index.items():
    answer_words[idx] = w
    
if run_on == 'cuda':
    vqa_net = torch.nn.DataParallel(Net(num_tokens), device_ids=[0])
    vqa_net.load_state_dict(saved_state["weights"])
    vqa_net = vqa_net.to(device)
else:
    vqa_net = Net(num_tokens)
    state_dict = saved_state["weights"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # remove `module.` if it exists
        new_state_dict[name] = v
    vqa_net.load_state_dict(new_state_dict)
    vqa_net = vqa_net.to(device)


# %%
 # for visualization to convert indices to tokens for questions
question_words = ["unk"] * num_tokens
for w, idx in token_to_index.items():
    question_words[idx] = w

# %% [markdown]
# Let's modify the VQA model to use pytorch-resnet. Our model will be called `vqa_resnet`.

# %%
class ResNetLayer4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.r_model = resnet.resnet152(pretrained=True)
        self.r_model.eval()
        self.r_model.to(device)

    def forward(self, x):
        x = self.r_model.conv1(x)
        x = self.r_model.bn1(x)
        x = self.r_model.relu(x)
        x = self.r_model.maxpool(x)
        x = self.r_model.layer1(x)
        x = self.r_model.layer2(x)
        x = self.r_model.layer3(x)
        return self.r_model.layer4(x)

class VQA_Resnet_Model(Net):
    def __init__(self, embedding_tokens):
        super().__init__(embedding_tokens)
        self.resnet_layer4 = ResNetLayer4()

    def forward(self, v, q, q_len):
        q = self.text(q, list(q_len.data))
        v = self.resnet_layer4(v)

        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)

        a = self.attention(v, q)
        v = apply_attention(v, a)

        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)
        return answer
    
if run_on == 'cuda':
    vqa_resnet = VQA_Resnet_Model(vqa_net.module.text.embedding.num_embeddings)
    # `device_ids` contains a list of GPU ids which are used for parallelization supported by `DataParallel`
    vqa_resnet = torch.nn.DataParallel(vqa_resnet, device_ids=[0])
else:
    vqa_resnet = VQA_Resnet_Model(vqa_net.text.embedding.num_embeddings)



# saved vqa model's parameters
partial_dict = vqa_net.state_dict()

state = vqa_resnet.state_dict()
state.update(partial_dict)
vqa_resnet.load_state_dict(state)

vqa_resnet.to(device)
vqa_resnet.eval()

# This is original VQA model without resnet. Removing it, since we do not need it
del vqa_net

# this is necessary for the backpropagation of RNNs models in eval mode
torch.backends.cudnn.enabled = False

# %% [markdown]
# # Input Utilities
# 
# Now we will need some utility functions for the inputs of our model. 
# 
# Let's start off with our image input transform function. We will separate out the normalization step from the transform in order to view the original image.

# %%
image_size = 448  # scale image to given size and center
central_fraction = 1.0

transform = get_transform(image_size, central_fraction=central_fraction)
transform_normalize = transform.transforms.pop()

# %% [markdown]
# Now for the input question, we will need an encoding function (to go from words -> indices):

# %%
def encode_question(question):
    """ Turn a question into a vector of indices and a question length """
    question_arr = question.lower().split()
    vec = torch.zeros(len(question_arr), device=device).long()
    for i, token in enumerate(question_arr):
        index = token_to_index.get(token, 0)
        vec[i] = index
    return vec, torch.tensor(len(question_arr), device=device)

# %% [markdown]
# # Baseline Inputs 

# %% [markdown]
# The insights API utilises captum's attribution API under the hood, hence we will need a baseline for our inputs. A baseline is (typically) a neutral output to reference in order for our attribution algorithm(s) to understand which features are important in making a prediction (this is very simplified explanation, 'Remark 1' in the [Integrated Gradients paper](https://arxiv.org/pdf/1703.01365.pdf) has an excellent explanation on why they must be utilised).
# 
# For images and for the purpose of this tutorial, we will let this baseline be the zero vector (a black image).

# %%
def baseline_image(x):
    return x * 0

# %% [markdown]
# For sentences, as done in the multi-modal VQA tutorial, we will use a sentence composed of padded symbols.
# 
# We will also require to pass our model through the [`configure_interpretable_embedding_layer`](https://captum.ai/api/utilities.html?highlight=configure_interpretable_embedding_layer#captum.attr._models.base.configure_interpretable_embedding_layer) function, which separates the embedding layer and precomputes word embeddings. To put it simply, this function allows us to precompute and give the embedding vectors directly to our model, which will allow us to reference the words associated to particular embeddings (for visualization purposes).

# %%
if run_on == 'cuda':
    interpretable_embedding = configure_interpretable_embedding_layer(
        vqa_resnet, "module.text.embedding")
else:
    interpretable_embedding = configure_interpretable_embedding_layer(
        vqa_resnet, "text.embedding")


PAD_IND = token_to_index["pad"]
token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)

def baseline_text(x):
    seq_len = x.size(0)
    ref_indices = token_reference.generate_reference(seq_len, device=device).unsqueeze(
        0
    )
    return interpretable_embedding.indices_to_embeddings(ref_indices).squeeze(0)

def input_text_transform(x):
    return interpretable_embedding.indices_to_embeddings(x)

# %% [markdown]
# # Using the Insights API
# 
# Finally we have reached the relevant part of the tutorial.
# 
# First let's create a utility function to allow us to pass data into the insights API. This function will essentially produce `Batch` objects, which tell the insights API what your inputs, labels and any additional arguments are.

# %%
def vqa_dataset(image, questions, targets):
    img = Image.open(image).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    for question, target in zip(questions, targets):
        q, q_len = encode_question(question)

        q = q.unsqueeze(0)
        q_len = q_len.unsqueeze(0)

        target_idx = answer_to_index[target]

        yield Batch(
            inputs=(img, q), labels=(target_idx,), additional_args=q_len
        )
    

# %% [markdown]
# Let's create our `AttributionVisualizer`, to do this we need the following:
# 
# - A score function, which tells us how to interpret the model's output vector
# - Description of the input features given to the model
# - The data to visualize (as described above)
# - Description of the output (the class names), in our case this is our answer words

# %% [markdown]
# In our case, we want to produce a single answer output via softmax

# %%
def score_func(o):
    return F.softmax(o, dim=1)

# %% [markdown]
# The following function will convert a sequence of question indices to the associated question words for visualization purposes. This will be provided to the `TextFeature` object to describe text features.

# %%
def itos(input):
    return [question_words[int(i)] for i in input.squeeze(0)]

# %% [markdown]
# Let's define some dummy data to visualize using the function we declared earlier.

# %%
dataset = vqa_dataset("./img/vqa/siamese.jpg", 
    ["what is on the picture",
    "what color is the cat",
    "where color are the cat eyes" ],
    ["cat", "white and brown", "blue"]
)    

# %% [markdown]
# Now let's describe our features. Each feature requires an input transformation function and a set of baselines. As described earlier, we will use the black image for the image baseline and a padded sequence for the text baseline.
# 
# The input image will be transformed via our normalization transform (`transform_normalize`).
# Our input text will need to be transformed into embeddings, as it is a sequence of indices. Our model only accepts embeddings as input, as we modified the model with `configure_interpretable_embedding_layer` earlier.
# 
# We also need to provide how the input text should be transformed in order to be visualized, which will be accomplished through the `itos` function, as described earlier.

# %%
features = [
    ImageFeature(
        "Picture",
        input_transforms=[transform_normalize],
        baseline_transforms=[baseline_image],
    ),
    TextFeature(
        "Question",
        input_transforms=[input_text_transform],
        baseline_transforms=[baseline_text],
        visualization_transform=itos,
    ),
]

# %% [markdown]
# An addition to the code was made to create perturbations for both text and images, enabling the evaluation of different explanation methods on sensitivity.

# %%
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
from nltk.corpus import wordnet
from itertools import chain
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


def perturb_image(image, noise_level=0.1, device='cpu'):
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image).to(device)
    
    noise = torch.randn(image.size(), device=device) * noise_level
    perturbed_image = image + noise
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  
    
    return perturbed_image

def get_synonyms(word):
    synonyms = wordnet.synsets(word)
    return set(chain.from_iterable([word.lemma_names() for word in synonyms]))

def perturb_text(text, perturbation_rate=0.1):

    words = text.split()
    num_perturb = int(len(words) * perturbation_rate)
    indices = random.sample(range(len(words)), num_perturb)
    
    for i in indices:
        synonyms = get_synonyms(words[i])
        if synonyms:
            words[i] = random.choice(list(synonyms))
    
    perturbed_text = ' '.join(words)
    return perturbed_text


# %%
def calculate_aopc_comprehensiveness(model, transformed_inputs, target, attributions, additional_args, steps=5):
    device = next(model.parameters()).device
    original_pred = model(*transformed_inputs, additional_args).max(dim=1).values
    
    scores = torch.zeros(steps, len(original_pred), device='cpu')
    
    for k in range(1, steps + 1):
        perturbed_inputs = []
        for i, input_tensor in enumerate(transformed_inputs):
            # Flatten the attribution tensor and get top k indices
            top_k_indices = attributions[i].view(-1).argsort(descending=True)[:k]
            # Clone the input tensor to avoid modifying the original tensor
            perturbed_input = input_tensor.clone()
            # Flatten the tensor, set top k indices to 0, and reshape back to original shape
            perturbed_input.view(-1)[top_k_indices] = 0
            perturbed_inputs.append(perturbed_input)

        perturbed_inputs = [pi.to(device) for pi in perturbed_inputs]
        
        perturbed_pred = model(*perturbed_inputs, additional_args).max(dim=1).values
        scores[k-1] = (original_pred - perturbed_pred).cpu() 

        torch.cuda.empty_cache()
    
    return scores.mean(dim=0).item()



import torch

def calculate_aopc_sufficiency(model, transformed_inputs, target, attributions, additional_args, steps=5):
    device = next(model.parameters()).device
    original_pred = model(*transformed_inputs, additional_args).max(dim=1).values
    
    scores = torch.zeros(steps, len(original_pred), device='cpu')
    
    for k in range(1, steps + 1):
        perturbed_inputs = []
        for i, input_tensor in enumerate(transformed_inputs):
            # Flatten the attribution tensor and get top k indices
            top_k_indices = attributions[i].view(-1).argsort(descending=True)[:k]
            # Create a mask with top k indices set to 0
            mask = torch.ones_like(input_tensor).view(-1)
            mask[top_k_indices] = 0
            # Clone the input tensor and apply the mask
            perturbed_input = input_tensor.clone()
            perturbed_input.view(-1)[mask.bool()] = 0
            perturbed_inputs.append(perturbed_input)

        perturbed_inputs = [pi.to(device) for pi in perturbed_inputs]

        perturbed_pred = model(*perturbed_inputs, additional_args).max(dim=1).values
        scores[k-1] = perturbed_pred.cpu() 
        
        torch.cuda.empty_cache()
    
    return scores.mean(dim=0).item()


def create_human_rationale_image(image_shape, important_regions, Test=True):
    if Test:
        human_rationale_image = torch.ones(image_shape)
    else:    
         human_rationale_image = torch.zeros(image_shape)
         for box in important_regions:
            x_start, x_end, y_start, y_end = box
            human_rationale_image[:, x_start:x_end, y_start:y_end] = 1

    return human_rationale_image


def evaluate_plausibility(attributions, human_rationale, threshold=0.5):
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0

    device = attributions[0].device
    human_rationale = [rationale.to(device) for rationale in human_rationale]

    for i, (attribution, rationale) in enumerate(zip(attributions, human_rationale)):
            
        if attribution.shape != rationale.shape:
            # Adjust shapes to match
            if len(rationale.shape) < len(attribution.shape):
                rationale = rationale.unsqueeze(0)  # Add batch dimension
            if rationale.shape[-1] != attribution.shape[-1]:
                rationale = rationale.unsqueeze(-1).expand_as(attribution)  # Match the last dimension

        binary_attributions = (attribution > threshold).float()
        true_positive = (binary_attributions * rationale).sum().item()
        false_positive = (binary_attributions * (1 - rationale)).sum().item()
        false_negative = ((1 - binary_attributions) * rationale).sum().item()

        total_true_positive += true_positive
        total_false_positive += false_positive
        total_false_negative += false_negative

    precision = total_true_positive / (total_true_positive + total_false_positive) if (total_true_positive + total_false_positive) != 0 else 0
    recall = total_true_positive / (total_true_positive + total_false_negative) if (total_true_positive + total_false_negative) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }








# %% [markdown]
# Let's define our AttributionVisualizer object with the above parameters and our `vqa_resnet` model. 

# %%
visualizer = AttributionVisualizer(
    models=[vqa_resnet],
    score_func=score_func,
    features=features,
    dataset=dataset,
    classes=answer_words,
)

# %% [markdown]
# And now we can visualize the outputs produced by the model.
# 
# Insights allows [different attribution methods](https://captum.ai/docs/algorithms) to be chosen. By default, [integrated gradients](https://captum.ai/api/integrated_gradients) is selected.

# %%
visualizer.serve()

# %% [markdown]
# Now that the visualizer is displayed on localhost, we can run explanations and obtain the necessary information to evaluate these models. In this first section, a sensitivity test was conducted to observe how perturbations impact the explanation and to determine the magnitude of perturbation required to produce an incorrect explanation.

# %%
from captum.attr import (Deconvolution,DeepLift,FeatureAblation,GuidedBackprop,InputXGradient,IntegratedGradients,Occlusion,Saliency,)
import pandas as pd
from tabulate import tabulate

SUPPORTED_ATTRIBUTION_METHODS = [Deconvolution,DeepLift,GuidedBackprop,InputXGradient,IntegratedGradients,Saliency,FeatureAblation,Occlusion,]
ATTRIBUTION_NAMES_TO_METHODS = {
    cls.get_name(): cls  # type: ignore
    for cls in SUPPORTED_ATTRIBUTION_METHODS
}

def calculate_attributions(model, inputs, target, baselines, additional_args, xai_model, selected_arguments):
    xai = ATTRIBUTION_NAMES_TO_METHODS[xai_model](model)
    if xai_model in ['IntegratedGradients', 'FeatureAblation', 'Occlusion']:
        attributions = xai.attribute.__wrapped__(xai, inputs=inputs, additional_forward_args=additional_args, target=target, **selected_arguments)
    else:
        attributions = xai.attribute.__wrapped__(xai, inputs=inputs, additional_forward_args=additional_args, target=target)
    return attributions

modality_attributions = visualizer.get_attributions()
 
selected_arguments= visualizer.get_insights_config()['selected_arguments']    
xai_model = visualizer.get_insights_config()['selected_method']
print(xai_model) 


# %% [markdown]
# Here is the code for calculating sensitivity by separating the picture and question, and creating individual perturbations for these inputs.

# %%
dataset = vqa_dataset("./img/vqa/siamese.jpg", 
    ["what is on the picture",
    "what color is the cat",
    "what color are the cat eyes" ],
    ["cat", "white and brown", "blue"]
)  

human_rationale_text = (torch.tensor([0,0,0,0,1]), torch.tensor([0,1,0,0,1]), torch.tensor([0,1,0,0,1,1]))  # Example human rationale for image


results = []
cont = 0



for batch in dataset:
    original_inputs = batch.inputs
    original_additional_args = batch.additional_args
    target = batch.labels
    
    noise_level = 0.1
    perturbation_rate = 0.1
    
    # Calculate original attributions
    (original_predicted_scores, original_baselines, transformed_inputs,) = visualizer.attribution_calculation.calculate_predicted_scores(original_inputs, original_additional_args, vqa_resnet)
    original_attributions = calculate_attributions(vqa_resnet, transformed_inputs, target, None, original_additional_args, xai_model, selected_arguments)
    original_net_contrib = visualizer.attribution_calculation.calculate_net_contrib(original_attributions)

    # Clear unused variables and cache to free GPU memory
    del original_baselines
    torch.cuda.empty_cache()
    
    original_label = original_predicted_scores[0].label
    
    # Ensure inputs and attributions are properly dimensioned
    inputs_image = transformed_inputs[0].unsqueeze(0)  # Add batch dimension if needed
    inputs_text = transformed_inputs[1].unsqueeze(0)   # Add batch dimension if needed
    inputs = (inputs_image, inputs_text)

    # AOPC Comprehensiveness
    aopc_comprehensiveness_score = calculate_aopc_comprehensiveness(vqa_resnet, transformed_inputs, target, original_attributions, original_additional_args)
    print(f"AOPC Comprehensiveness Score: {aopc_comprehensiveness_score}")

    # AOPC Sufficiency
    aopc_sufficiency_score = calculate_aopc_sufficiency(vqa_resnet, transformed_inputs, target, original_attributions, original_additional_args)
    print(f"AOPC Sufficiency Score: {aopc_sufficiency_score}")

    # Plausibility Evaluation
    #This plausibility checks needs to be done for each image and text pair in the dataset with the human rationale. For this example we dont know the regions of the images that have the highest importance, so we will use the entire image as the human rationale.

    important_region = [(87, 137, 87, 137), (50, 100, 50, 100)]  # Example important region (center 50x50 region) (NOT USED IN THIS EXAMPLE) 
    image_shape = original_inputs[0].shape[1:]  # Example image shape (C, H, W)
    human_rationale_image = create_human_rationale_image(image_shape, important_region)
    human_rationale = (human_rationale_image, human_rationale_text[cont])
    plausibility_scores = evaluate_plausibility(original_attributions, human_rationale)
    print(f"Plausibility Scores: {plausibility_scores}")
    cont += 1
    
    #Sensitivity Test
    while True:
        # Perturb image
        perturbed_image = perturb_image(original_inputs[0][0], noise_level=noise_level, device=run_on)
        perturbed_image = perturbed_image.unsqueeze(0).to(device)

        # Perturb text
        original_question = ' '.join(itos(original_inputs[1][0]))
        perturbed_question = perturb_text(original_question, perturbation_rate=perturbation_rate)
        perturbed_question_vec, perturbed_question_len = encode_question(perturbed_question)
        perturbed_question_vec = perturbed_question_vec.unsqueeze(0)
        perturbed_question_len = perturbed_question_len.unsqueeze(0)

        perturbed_inputs = (perturbed_image, perturbed_question_vec)
        perturbed_additional_args = perturbed_question_len

        # Calculate new attributions with perturbed inputs
        (perturbed_predicted_scores, perturbed_baselines, perturbed_transformed_inputs,) = visualizer.attribution_calculation.calculate_predicted_scores(perturbed_inputs, perturbed_additional_args, vqa_resnet)
        perturbed_attributions = calculate_attributions(vqa_resnet, perturbed_transformed_inputs, target, None, perturbed_additional_args, xai_model, selected_arguments)
        perturbed_net_contrib = visualizer.attribution_calculation.calculate_net_contrib(perturbed_attributions)

        # Clear unused variables and cache to free GPU memory
        del perturbed_baselines, perturbed_transformed_inputs
        torch.cuda.empty_cache()

        # Compare original and perturbed attributions
        perturbed_label = perturbed_predicted_scores[0].label
        prediction_consistency = original_label == perturbed_label
        comparison = np.array(original_net_contrib) - np.array(perturbed_net_contrib)

        # Store the results
        results.append({
            'noise_level': noise_level,
            'perturbation_rate': perturbation_rate,
            'original_label': original_label,
            'perturbed_label': perturbed_label,
            'prediction_consistency': prediction_consistency,
            'original_net_contrib [Picture, Question]': str(original_net_contrib),
            'perturbed_net_contrib [Picture, Question]': str(perturbed_net_contrib),
            'comparison [Picture, Question]': str(comparison)
        })

        # Check for consistency
        if not prediction_consistency:
            break
        else:
            noise_level += 0.2
            perturbation_rate += 0.1

# Create a DataFrame from the results
results_df = pd.DataFrame(results)
print('Sensitivity test results:')
print(tabulate(results_df, headers='keys', tablefmt='psql'))

del perturbed_image, original_question, perturbed_question, perturbed_question_vec, perturbed_question_len, perturbed_inputs, perturbed_additional_args
torch.cuda.empty_cache()



# %%
# show a screenshot if using notebook non-interactively
import IPython.display
IPython.display.Image(filename='img/captum_insights_vqa.png')

# %% [markdown]
# Finally, since we are done with visualization, we will revert the change to the model we made with `configure_interpretable_embedding_layer`. To do this, we will invoke the `remove_interpretable_embedding_layer` function. Uncomment the line below to execute the cell.

# %%
# remove_interpretable_embedding_layer(vqa_resnet, interpretable_embedding)


