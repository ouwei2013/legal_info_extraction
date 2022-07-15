from setuptools import setup

setup(
    name='legal_info_extraction',
    version='0.1.0',    
    author='wei ou',
    packages=['legal_info_extraction'],
    install_requires=['spacy==3.3.0',
                      'spacy[transformers,cuda111]',
                      'numpy','bs4','regex'                   
                      ]
)