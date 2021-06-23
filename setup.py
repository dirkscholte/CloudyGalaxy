import setuptools

setuptools.setup(
    name="CloudyGalaxy",
    version="0.1",
    author="Dirk Scholte",
    author_email="dirk.scholte.20@ucl.ac.uk",
    description="Python package to generate photoionization models to produce galaxy emission lines.",
    packages=["CloudyGalaxy", "numpy", "scipy", "pyCloudy", "math", "astropy"]
)
