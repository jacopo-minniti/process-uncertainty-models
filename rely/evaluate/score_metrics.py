import torch
import torch.nn.functional as F
from tqdm import tqdm
import re
from collections import Counter
import math
import numpy as np
from rely.utils import load_dataset, save_dataset

def calculate_semantic_isotropy(embeddings: torch.Tensor) -> float:
    """
    Calculates the semantic isotropy score for a given set of embeddings.
    
    The score is the von Neumann entropy of the cosine similarity matrix
    of the normalized embeddings.
    
    Args:
        embeddings: A tensor of shape [K, D], where K is the number of
                    completions and D is the embedding dimension.
    
    Returns:
        The calculated semantic isotropy score as a float.
    """
    # Ensure calculations are done without tracking gradients
    with torch.no_grad():
        # Step 0: Handle edge cases where a score cannot be computed
        if not isinstance(embeddings, torch.Tensor) or embeddings.shape[0] < 2:
            return -1

        # Step 1: Normalize the embeddings to have unit L2 norm
        # This projects the embeddings onto the unit sphere
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        # Step 2: Compute the Cosine Kernel Matrix
        # For normalized vectors, this is equivalent to a matrix multiplication
        kernel_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)
        
        # Step 3: Calculate the von Neumann Entropy of the kernel matrix
        # This requires finding the eigenvalues of the matrix.
        # We use .eigvalsh() as the kernel matrix is symmetric, which is more stable.
        try:
            # Clamp the matrix to be perfectly symmetric and add a small identity
            # matrix for numerical stability before finding eigenvalues.
            k = (kernel_matrix + kernel_matrix.T) / 2
            k = k + 1e-6 * torch.eye(k.shape[0], device=k.device)
            eigenvalues = torch.linalg.eigvalsh(k)
        except Exception as e:
            print(f"Error calculating eigenvalues: {e}")
            # If eigenvalue computation fails, return a default score
            return -1

        # Filter out non-positive eigenvalues that can result from numerical instability
        positive_eigenvalues = eigenvalues[eigenvalues > 1e-6]
        if positive_eigenvalues.numel() == 0:
            return -1

        # The von Neumann entropy formula is -Tr(p * log(p)), where p is the density
        # matrix. In terms of eigenvalues, this is -sum(eig * log(eig)).
        entropy = -torch.sum(positive_eigenvalues * torch.log(positive_eigenvalues))
        
        return entropy.item()

def calculate_entropy_from_completions(completions):
    """
    Calculates the entropy of the categorical distribution of the last uppercase letter in parentheses (e.g., (A))
    from a list of completion strings.
    Args:
        completions: List of strings.
    Returns:
        Entropy (float) of the distribution. Returns -1 if no valid letters are found or if 5+ invalid letters.
    """
    if not isinstance(completions, list) or len(completions) == 0:
        return -1
    # Regex to find the last (A), (B), ... in each string
    pattern = re.compile(r"\(([A-Z])\)")
    letters = []
    invalid_count = 0
    for c in completions:
        matches = pattern.findall(c)
        if matches:
            letters.append(matches[-1])
        else:
            invalid_count += 1
            # Return -1 if we have 5 or more invalid letters
            if invalid_count >= 5:
                return -1
    if not letters:
        return -1
    counts = Counter(letters)
    total = sum(counts.values())
    probs = [count / total for count in counts.values()]
    entropy = -sum(p * math.log(p) for p in probs)
    return entropy

def calculate_hard_label(completions, correct_answer):
    """
    Calculates hard_label which is 1 if at least one completion is correct, otherwise 0.
    
    Args:
        completions: List of completion strings
        correct_answer: The correct answer letter (e.g., 'A', 'B', 'C', 'D')
    
    Returns:
        Hard label (int): 1 if at least one completion is correct, 0 otherwise
    """
    if not isinstance(completions, list) or len(completions) == 0:
        return -1
    
    # Regex to find the last (A), (B), ... in each string
    pattern = re.compile(r"\(([A-Z])\)")
    
    for completion in completions:
        matches = pattern.findall(completion)
        if matches:
            predicted_answer = matches[-1]
            if predicted_answer == correct_answer:
                return 1
    
    return 0

def calculate_soft_label(completions, correct_answer):
    """
    Calculates soft_label which is the percentage of correct completions.
    
    Args:
        completions: List of completion strings
        correct_answer: The correct answer letter (e.g., 'A', 'B', 'C', 'D')
    
    Returns:
        Soft label (float): Percentage of correct completions (0.0 to 1.0)
    """
    if not isinstance(completions, list) or len(completions) == 0:
        return -1
    
    # Regex to find the last (A), (B), ... in each string
    pattern = re.compile(r"\(([A-Z])\)")
    correct_count = 0
    valid_count = 0
    
    for completion in completions:
        matches = pattern.findall(completion)
        if matches:
            valid_count += 1
            predicted_answer = matches[-1]
            if predicted_answer == correct_answer:
                correct_count += 1
    
    if valid_count == 0:
        return -1
    
    return correct_count / valid_count

def calculate_variance(completions, correct_answer):
    """
    Calculates variance of zeros/ones based on correctness.
    
    Args:
        completions: List of completion strings
        correct_answer: The correct answer letter (e.g., 'A', 'B', 'C', 'D')
    
    Returns:
        Variance (float): Variance of the binary correctness values
    """
    if not isinstance(completions, list) or len(completions) == 0:
        return -1
    
    # Regex to find the last (A), (B), ... in each string
    pattern = re.compile(r"\(([A-Z])\)")
    correctness_values = []
    
    for completion in completions:
        matches = pattern.findall(completion)
        if matches:
            predicted_answer = matches[-1]
            # 1 if correct, 0 if incorrect
            correctness_values.append(1 if predicted_answer == correct_answer else 0)
    
    if len(correctness_values) == 0:
        return -1
    
    # Calculate variance
    correctness_array = np.array(correctness_values)
    variance = np.var(correctness_array)
    
    return variance

def score(input_file, output_file, dataset_percentage=1.0):
    """
    Process the dataset and add entropy, hard_label, soft_label, and variance scores.
    
    Args:
        input_file (str): Path to the input dataset file (.pt format)
        output_file (str): Path to save the processed dataset (.pt format)
        dataset_percentage (float): Percentage of dataset to process (0.0 to 1.0)
    
    Returns:
        dict: Statistics about the processed data
    """
    print(f"Loading dataset from {input_file}...")
    
    # Load the dataset
    data = load_dataset(input_file)
    
    # Take only a percentage of the dataset
    num_items_to_process = int(len(data) * dataset_percentage)
    data = data[:num_items_to_process]
    print(f"Processing {len(data)} items ({dataset_percentage*100:.1f}% of dataset)")
    
    # Process each item and add entropy
    processed_data = []
    
    for i, item in enumerate(tqdm(data, desc="Processing items")):
        # Create a copy of the item to avoid modifying the original
        processed_item = item.copy()
        
        # Calculate entropy from completions
        if 'completions' in item:
            entropy_score = calculate_entropy_from_completions(item['completions'])
            processed_item['entropy'] = entropy_score
            
            # Get correct answer from the data
            correct_answer = None
            if 'solution' in item:
                correct_answer = item['solution']
            
            # Calculate additional metrics if we have the correct answer
            if correct_answer is not None:
                hard_label = calculate_hard_label(item['completions'], correct_answer)
                soft_label = calculate_soft_label(item['completions'], correct_answer)
                variance = calculate_variance(item['completions'], correct_answer)
                
                processed_item['hard_label'] = hard_label
                processed_item['soft_label'] = soft_label
                processed_item['variance'] = variance
                processed_item['correct_answer'] = correct_answer
            
            # Only add items with valid entropy scores (not -1)
            if entropy_score != -1:
                processed_data.append(processed_item)
        else:
            print(f"Warning: Item {i} has no 'completions' field")
            continue
    
    # Save the processed data
    print(f"Saving processed data to {output_file}...")
    save_dataset(processed_data, output_file)
    
    # Calculate statistics
    stats = {}
    
    # Print some statistics
    valid_entropies = [item['entropy'] for item in processed_data if item['entropy'] != -1]
    if valid_entropies:
        stats['entropy'] = {
            'mean': sum(valid_entropies) / len(valid_entropies),
            'min': min(valid_entropies),
            'max': max(valid_entropies),
            'valid_items': len(valid_entropies),
            'total_items': len(processed_data)
        }
        print(f"Entropy statistics:")
        print(f"  Mean: {stats['entropy']['mean']:.4f}")
        print(f"  Min: {stats['entropy']['min']:.4f}")
        print(f"  Max: {stats['entropy']['max']:.4f}")
        print(f"  Valid items: {stats['entropy']['valid_items']}/{stats['entropy']['total_items']}")
    else:
        print("No valid entropy scores found")
    
    # Print statistics for new metrics
    valid_hard_labels = [item['hard_label'] for item in processed_data if 'hard_label' in item and item['hard_label'] != -1]
    if valid_hard_labels:
        stats['hard_label'] = {
            'mean': sum(valid_hard_labels) / len(valid_hard_labels),
            'correct_predictions': sum(valid_hard_labels),
            'total_predictions': len(valid_hard_labels),
            'accuracy': sum(valid_hard_labels) / len(valid_hard_labels)
        }
        print(f"\nHard Label statistics:")
        print(f"  Mean: {stats['hard_label']['mean']:.4f}")
        print(f"  Correct predictions: {stats['hard_label']['correct_predictions']}/{stats['hard_label']['total_predictions']} ({stats['hard_label']['accuracy']*100:.1f}%)")
    
    valid_soft_labels = [item['soft_label'] for item in processed_data if 'soft_label' in item and item['soft_label'] != -1]
    if valid_soft_labels:
        stats['soft_label'] = {
            'mean': sum(valid_soft_labels) / len(valid_soft_labels),
            'min': min(valid_soft_labels),
            'max': max(valid_soft_labels)
        }
        print(f"\nSoft Label statistics:")
        print(f"  Mean: {stats['soft_label']['mean']:.4f}")
        print(f"  Min: {stats['soft_label']['min']:.4f}")
        print(f"  Max: {stats['soft_label']['max']:.4f}")
    
    valid_variances = [item['variance'] for item in processed_data if 'variance' in item and item['variance'] != -1]
    if valid_variances:
        stats['variance'] = {
            'mean': sum(valid_variances) / len(valid_variances),
            'min': min(valid_variances),
            'max': max(valid_variances)
        }
        print(f"\nVariance statistics:")
        print(f"  Mean: {stats['variance']['mean']:.4f}")
        print(f"  Min: {stats['variance']['min']:.4f}")
        print(f"  Max: {stats['variance']['max']:.4f}")
    
    return stats
