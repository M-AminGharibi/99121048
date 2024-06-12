import torch
from sklearn.metrics import f1_score

# Assume these are the true grades
true_grades = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1])

# Define the class for failing the course
fail_class = 1
pass_class = 0

# Case 1: Random guesses
# Generate random predictions (0 or 1)
random_predictions = torch.randint(0, 2, true_grades.shape)

# Calculate F1-score for random guesses
f1_random = f1_score(true_grades.numpy(), random_predictions.numpy(), pos_label=fail_class)
print(f'F1-score for random guesses: {f1_random:.4f}')

# Case 2: Model that always predicts passing
# Generate predictions that always predict passing
always_pass_predictions = torch.full(true_grades.shape, pass_class)

# Calculate F1-score for always pass model
f1_always_pass = f1_score(true_grades.numpy(), always_pass_predictions.numpy(), pos_label=fail_class)
print(f'F1-score for always pass model: {f1_always_pass:.4f}')
