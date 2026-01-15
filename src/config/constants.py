"""
Model and training constants.

These values are extracted from inline magic numbers to improve
configurability and documentation.
"""

# Model Architecture
MESSAGE_PASSING_RATIO = 0.3  # Fraction of hidden_dim for message passing (x_other)
DEFAULT_ATTENTION_TEMPERATURE = 1.0  # Softmax temperature for attention pooling

# Stereochemistry
TETRAHEDRAL_MAGNITUDE_SCALE = 3.0  # Divisor for tanh scaling of tetrahedral features

# Training
GRADIENT_CLIP_MAX_NORM = 1.0  # Maximum gradient norm for clipping
DEFAULT_EVIDENTIAL_LAMBDA = 1.0  # Default regularization for evidential loss
