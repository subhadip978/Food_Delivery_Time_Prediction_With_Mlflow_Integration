# from zenml.steps import BaseParameters

from dataclasses import dataclass

@dataclass
class ModelNameConfig:
    model_name: str = "linearregression"

# Example usage
config = ModelNameConfig()
print(config.model_name)  # Output: randomforest
