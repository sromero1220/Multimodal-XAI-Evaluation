# evaluation.py

from scipy.stats import spearmanr
from textstat import flesch_kincaid_grade
import torch

def sensitivity_to_perturbations(model, inputs, attributions, perturbation_factor=0.1):
    perturbed_inputs = inputs.clone()
    important_features = attributions > attributions.mean()
    perturbed_inputs[important_features] += perturbation_factor
    original_output = model(inputs)
    perturbed_output = model(perturbed_inputs)
    return torch.norm(original_output - perturbed_output).item()

def agreement_with_human_judgment(attributions, human_judgment):
    agreement = sum([1 if a in human_judgment else 0 for a in attributions]) / len(attributions)
    return agreement

def explanation_complexity(text_explanation):
    return flesch_kincaid_grade(text_explanation)


###########################################################################################################

def feature_importance_consistency(attributions, model_weights):
    importance_scores = attributions.detach().cpu().numpy()
    weights = model_weights.detach().cpu().numpy()
    return spearmanr(importance_scores, weights).correlation

def perturbation_test(attributions, inputs, model, perturbation_factor=0.1):
    perturbed_inputs = [input.clone() for input in inputs]
    
    # Perturb image inputs based on attributions
    image_attributions = attributions[0]
    image_perturbation = image_attributions * perturbation_factor
    perturbed_inputs[0] += image_perturbation
    
    # Perturb text inputs based on attributions
    text_attributions = attributions[1]
    text_perturbation = text_attributions * perturbation_factor
    perturbed_inputs[1] += text_perturbation

    original_output = model(*inputs)
    perturbed_output = model(*perturbed_inputs)
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

