from setuptools import setup, find_packages

setup(
    name='automl_pipeline',
    version='1.0.0',
    description='Automated Machine Learning Pipeline for Rapid Prototyping',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'scikit-learn',
        'pandas',
        'PyYAML',
        'xgboost',
        'lightgbm'
    ],
)