from setuptools import setup, find_packages

setup(
    name="aorticarchgenerator",
    version="0.1",
    description="Aortic Arch Generator for the use of endovascular simulations",
    author="Lennart Karstensen",
    author_email="lennart.karstensen@ipa.fraunhofer.de",
    packages=find_packages(),
    install_requires=["pyvista", "scikit-image", "meshio", "numpy"],
)
