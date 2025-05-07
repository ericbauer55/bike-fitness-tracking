from setuptools import setup, find_packages

with open('app/README.md', 'r') as f:
    long_description = f.read()

# Ref: https://setuptools.pypa.io/en/latest/references/keywords.html

setup(
    name="ride_extractor",
    version="0.0.1",
    description="This package helps extract bike ride data from Strava-rendered GPX files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"":"app"}, 
    packages=find_packages(where="app"),
    author="Eric Bauer",
    author_email="eric.bauer55@gmail.com",
    license="MIT",
    install_requires=["gpxpy==1.6.2", "pandas==2.2.3"],
    extras_require={
        "dev":["pytest>=8.0"] # these will be installed when using "pip install tcgextract[dev]"
    },
    python_requires=">=3.12"

)

