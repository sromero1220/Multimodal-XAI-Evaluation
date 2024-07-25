# %% [markdown]
# # Captum Insights for Visual Question Answering

# %% [markdown]
# This notebook provides a simple example for the [Captum Insights API](https://captum.ai/docs/captum_insights), which is an easy to use API built on top of Captum that provides a visualization widget.
# 
# It is suggested to first read the multi-modal [tutorial](https://captum.ai/tutorials/Multimodal_VQA_Interpret) with VQA that utilises the `captum.attr` API. This tutorial will skip over a large chunk of details for setting up the VQA model.
# 
# As with the referenced tutorial, you will need the following installed on your machine:
# 
# - Python Packages: torchvision, PIL, and matplotlib
# - pytorch-vqa: https://github.com/Cyanogenoid/pytorch-vqa
# - pytorch-resnet: https://github.com/Cyanogenoid/pytorch-resnet
# - A pretrained pytorch-vqa model, which can be obtained from: https://github.com/Cyanogenoid/pytorch-vqa/releases/download/v1.0/2017-08-04_00.55.19.pth
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
# Now, we will import the necessary modules to run the code in this tutorial. Please make sure you have the [prerequisites to run captum](https://captum.ai/docs/getting_started), along with the pre-requisites to run this tutorial (as described in the first section).

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
dataset = vqa_dataset("./img/vqa/elephant.jpg", 
    ["what is on the picture",
    "what color is the elephant",
    "where is the elephant" ],
    ["elephant", "gray", "zoo"]
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

def perturb_image(image, noise_level=0.1):
    """
    Apply random noise to the image.
    
    :param image: PIL Image or tensor
    :param noise_level: float, the standard deviation of the noise
    :return: perturbed image as tensor
    """
    if isinstance(image, Image.Image):
        image = transforms.ToTensor()(image)
    
    noise = torch.randn(image.size()) * noise_level
    perturbed_image = image + noise
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure pixel values are within [0, 1]
    
    return perturbed_image

def get_synonyms(word):
    synonyms = wordnet.synsets(word)
    return set(chain.from_iterable([word.lemma_names() for word in synonyms]))

def perturb_text(text, perturbation_rate=0.1):
    """
    Replace words in the text with their synonyms.
    
    :param text: str, input text
    :param perturbation_rate: float, percentage of words to be replaced
    :return: perturbed text
    """
    words = text.split()
    num_perturb = int(len(words) * perturbation_rate)
    indices = random.sample(range(len(words)), num_perturb)
    
    for i in indices:
        synonyms = get_synonyms(words[i])
        if synonyms:
            words[i] = random.choice(list(synonyms))
    
    perturbed_text = ' '.join(words)
    return perturbed_text

# Example usage for image
image = Image.open("path/to/your/image.jpg")
perturbed_image = perturb_image(image, noise_level=0.1)
transforms.ToPILImage()(perturbed_image).show()

# Example usage for text
text = "This is an example text for perturbation."
perturbed_text = perturb_text(text, perturbation_rate=0.2)
print(perturbed_text)


# %% [markdown]
# Here, all the different functions for evaluating models on the multimodal dataset is coded.

# %%
from scipy.stats import spearmanr
from textstat import flesch_kincaid_grade
import torch
from captum.attr import IntegratedGradients


# Evaluation functions
def feature_importance_consistency(attributions, model_weights):
    # Summarize attributions to match the dimensionality of model weights
    summarized_attributions = torch.mean(torch.tensor(attributions), dim=0).detach().cpu().numpy()
    weights = model_weights.detach().cpu().numpy()
    return spearmanr(summarized_attributions, weights[:len(summarized_attributions)]).correlation

def perturbation_test(attributions, inputs, model, additional_args, perturbation_factor=0.1):
    perturbed_inputs = []
    for input in inputs:
        perturbed_input = []
        for element in input:
            if isinstance(element, torch.Tensor):
                perturbed_input.append(element.clone())
            else:
                perturbed_input.append(element)
        perturbed_inputs.append(perturbed_input)

    # Get the device of the inputs
    device = inputs[0][0].device

    # Convert attributions to tensors and move them to the same device as inputs
    image_attributions = torch.tensor([attr[0] for attr in attributions]).to(device)
    text_attributions = torch.tensor([attr[1] for attr in attributions]).to(device)

    # Debug prints
    print(f"Image attributions shape: {image_attributions.shape}")
    print(f"Text attributions shape: {text_attributions.shape}")

    # Ensure image_attributions matches the shape of the image input
    for i in range(len(perturbed_inputs)):
        batch_size, channels, height, width = perturbed_inputs[i][0].shape
        print(f"Perturbed input shape: {perturbed_inputs[i][0].shape}")
        image_perturbation = image_attributions.view(batch_size, channels, height, width) * perturbation_factor
        perturbed_inputs[i][0] += image_perturbation

        # Ensure text_attributions matches the shape of the question input
        sequence_length = perturbed_inputs[i][1].shape[1]
        text_perturbation = text_attributions.view(batch_size, sequence_length) * perturbation_factor
        perturbed_inputs[i][1] += text_perturbation

    original_output = model(*inputs, additional_args)
    perturbed_output = model(*[tuple(input) for input in perturbed_inputs], additional_args)
    return torch.norm(original_output - perturbed_output).item()


def readability_score(text):
    return flesch_kincaid_grade(text)

def stability_test(attributions, inputs, model, additional_args, noise_level=0.1):
    noisy_inputs = [input + noise_level * torch.randn_like(input) for input in inputs]
    noisy_attributions = visualizer.attribution_calculation.calculate_attribution(
        baselines=None,
        data=noisy_inputs,
        additional_forward_args=additional_args,
        label=None,
        attribution_method_name="IntegratedGradients",
        attribution_arguments={'n_steps': 25},
        model=model
    )
    return torch.norm(attributions - noisy_attributions).item()

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
visualizer.serve(debug=True)

# %% [markdown]
# Now that the visualizer is being displayed on localhost, when a explanation is run we have the requiered information to create evaluations for this models.

# %%
from captum.attr._utils.batching import _batched_generator


modality_attributions = visualizer.get_attributions()
# Inspect the extracted attributions
print("Extracted Attributions:")
for mod_attr in modality_attributions:
    print(mod_attr)
    
    
dataset = vqa_dataset("./img/vqa/elephant.jpg", 
    ["what is on the picture",
    "what color is the elephant",
    "where is the elephant" ],
    ["elephant", "gray", "zoo"]
)    
# inputs= []
# labels = []
# additional_args = []
# for batch in dataset:
#     inputs.append(batch.inputs)
#     labels.append(batch.labels)
#     additional_args.append(batch.additional_args)
# print(labels)

#labels = torch.tensor(labels)
# inputs= []
# labels = []
# additional_args = []
# for batch in dataset:
#     inputs.append(batch.inputs)
#     labels.append(batch.labels)
#     additional_args.append(batch.additional_args)
# print(inputs)

# labels = torch.tensor(labels)

# batch_data = next(iter(dataset))
# for (
#             inputs,
#             additional_forward_args,
#             label,
#         ) in _batched_generator(  # type: ignore
#             inputs=batch_data.inputs,
#             additional_forward_args=batch_data.additional_args,
#             target_ind=batch_data.labels,
#             internal_batch_size=1,  # should be 1 until we have batch label support
#         ):
#             print(inputs)
#             (predicted_scores, baselines, transformed_inputs,) = visualizer.attribution_calculation.calculate_predicted_scores(inputs, additional_forward_args,ATTRIBUTION_NAMES_TO_METHODS[xai_model])  
#             attrs_per_feature = visualizer.attribution_calculation.calculate_attribution(
#                     baselines,
#                     transformed_inputs,
#                     additional_forward_args,
#                     target,
#                     visualizer._config.attribution_method,
#                     visualizer._config.attribution_arguments,
#                     ATTRIBUTION_NAMES_TO_METHODS[xai_model],
#                 )
#             if target is None:
#                 target = (
#                     predicted_scores[0].index if len(predicted_scores) > 0 else None
#                 )
#             net_contrib = visualizer.attribution_calculation.calculate_net_contrib(attrs_per_feature)

batch_data = next(iter(dataset))

(inputs,additional_forward_args,label,)=_batched_generator(  # type: ignore
            inputs=batch_data.inputs,
            additional_forward_args=batch_data.additional_args,
            target_ind=batch_data.labels,
            internal_batch_size=1,  # should be 1 until we have batch label support
        )
print(visualizer.get_insights_config()['selected_arguments'])
xai_model = visualizer.get_insights_config()['selected_model']
print(visualizer.get_insights_config()['selected_method'])

(predicted_scores, baselines, transformed_inputs,) = visualizer.attribution_calculation.calculate_predicted_scores(inputs, additional_forward_args, xai_model)


# %%
# Evaluate explanations
model_weights = vqa_resnet.module.classifier.lin2.weight # Example for accessing model weights
faithfulness_score = feature_importance_consistency(modality_attributions, model_weights)
#perturbation_score = perturbation_test(attributions, inputs, vqa_resnet, additional_args)
readability = readability_score("Generated Explanation Text")  # Replace with actual text
#sensitivity_score = stability_test(attributions, inputs, vqa_resnet)

# Print or log evaluation results
print(f"Faithfulness Score: {faithfulness_score}")
#print(f"Perturbation Score: {perturbation_score}")
print(f"Readability Score: {readability}")
#print(f"Sensitivity Score: {sensitivity_score}")

# %%
# show a screenshot if using notebook non-interactively
import IPython.display
IPython.display.Image(filename='img/captum_insights_vqa.png')

# %% [markdown]
# Finally, since we are done with visualization, we will revert the change to the model we made with `configure_interpretable_embedding_layer`. To do this, we will invoke the `remove_interpretable_embedding_layer` function. Uncomment the line below to execute the cell.

# %%
# remove_interpretable_embedding_layer(vqa_resnet, interpretable_embedding)


