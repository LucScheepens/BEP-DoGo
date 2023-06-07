import torch
import torch.utils.data as data

class ClassProbabilitySampler(data.Sampler):
    def __init__(self, dataset, class_probabilities):
        self.dataset = dataset
        self.class_probabilities = class_probabilities

    def __iter__(self):
        indices = []
        num_samples = len(self.dataset)

        # Generate indices based on class probabilities
        for class_idx, class_prob in enumerate(self.class_probabilities):
            class_indices = [idx for idx in range(num_samples) if self.dataset[idx][1] == class_idx]
            num_class_samples = int(class_prob * num_samples)
            sampled_indices = torch.randperm(len(class_indices))[:num_class_samples]
            indices.extend([class_indices[idx] for idx in sampled_indices])

        return iter(indices)

    def __len__(self):
        return len(self.dataset)