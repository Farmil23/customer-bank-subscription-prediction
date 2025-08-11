"""
    setup.py file essential for packaging and distributing python project
"""

from setuptools import find_packages, setup
from typing import List

def get_requirements()->List[str]:
    """
        This functtion will return list of requirements
    """
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt', 'r') as file:
            # Raed lines from the file
            lines = file.readlines()
                
            ## process each line
            for line in lines:
                requirement = line.strip()
                
                ## ignore empty lines and -e .
                if requirement and requirement != "-e .":
                    requirement_lst.append(requirement)
                    
    except FileNotFoundError:
        print("Requirements.txt was not found")
        
    return requirement_lst
    
setup(
    name = "NetworkSecurity",
    version= "0.0.1",
    author = "Farhan kamil hermansyah",
    author_email="farmiljobs@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements()
)