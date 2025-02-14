import argparse
from pathlib import Path

from unitxt import add_to_catalog
from unitxt import get_from_catalog
from unitxt.templates import MultipleChoiceTemplate, Template

from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants


class CatalogManager:
    """
    Class to save datasets to the Unitxt local catalog.
    """

    def __init__(self, catalog_path: Path) -> None:
        """
        Initializes the DatasetSaver with the path to the local catalog.
        @param catalog_path: The path to the local catalog.

        @return: None
        """
        self.catalog_path = catalog_path

    def save_to_catalog(self, template: Template, name: str) -> None:
        """
        Saves the provided dataset to the local catalog.

        @param template: The Unitxt template for the dataset format.
        @param name: The desired name for the dataset in the catalog.

        @return: None
        """
        add_to_catalog(template, name, catalog_path=str(self.catalog_path), overwrite=True)
        print(f"Dataset saved successfully to local catalog: {self.catalog_path}")

    def load_from_catalog(self, name: str) -> Template:
        """
        Loads the dataset from the local catalog.

        @param name: The name of the dataset in the local catalog.

        @return: The dataset from the local catalog.
        """
        return get_from_catalog(name, catalog_path=str(self.catalog_path))


if __name__ == "__main__":
    # Define the path to your local catalog directory within your repository (replace with your actual path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog_path", type=str, default=TemplatesGeneratorConstants.CATALOG_PATH)
    parser.add_argument("--template_name", type=str, default="my_custom_qa_template")
    args = parser.parse_args()

    # Create an instance of the DatasetSaver
    catalog_manager = CatalogManager(args.catalog_path)

    # Define the MultipleChoiceTemplate (assuming your data adheres to this format)

    template = MultipleChoiceTemplate(
        input_format="The following are multiple choice questions (with answers) about {topic}"
                     ".\n\nQuestion: {question}\nChoose from {numerals}\nAnswers:\n{choices}\nAnswer:",
        target_field="answer",
        choices_separator="\n",
        enumerator="numerals",
        postprocessors=["processors.first_character"],
    )

    # Save the dataset using the saver
    catalog_manager.save_to_catalog(template, args.template_name)
    my_task = get_from_catalog(args.template_name, catalog_path=args.catalog_path)
