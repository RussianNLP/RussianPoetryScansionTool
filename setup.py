import setuptools
from pathlib import Path


def get_requirements():
    """Load requirements from requirements.txt."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip()
            for line in f.readlines()
            if line.strip() and not line.startswith(("#", "-"))
        ]
    return requirements


with open("README.md") as fh:
    long_description = fh.read()

setuptools.setup(
    name="russian_scansion",
    version="1.0.22",
    author="Ilya Koziev",
    author_email="inkoziev@gmail.com",
    description="Russian Poetry Scansion Tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Koziev/RussianPoetryScansionTool",
    packages=setuptools.find_packages(),
    package_data={'russian_scansion': [
        'models/word2lemma.pkl',
        'models/udpipe_syntagrus.model',
        'models/scansion_tool/scansion_tool.pkl',
        'models/accentuator/accents.pkl',
        'models/accentuator/pytorch_model.pth',
        'models/accentuator/config.json'
        ]},
    include_package_data=True,
    install_requires=get_requirements(),
)
