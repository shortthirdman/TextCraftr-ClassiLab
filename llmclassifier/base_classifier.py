from typing import Literal
from pydantic import BaseModel

def generate_classification_model(list_classes: list[str]):
    assert list_classes  # Ensure the list of classes is not empty

    class ClassificationOutput(BaseModel):
        category: Literal[tuple(list_classes)]

    return ClassificationOutput

# Example usage
if __name__ == "__main__":
    Categories = generate_classification_model(["Yes", "No"])
    categories = Categories(category="Yes")
    print(categories)