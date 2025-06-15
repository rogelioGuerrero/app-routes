from setuptools import setup, find_packages

setup(
    name="vrp_solver",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        'fastapi>=0.68.0',
        'uvicorn>=0.15.0',
        'pydantic>=1.8.0',
        'ortools>=9.0.0'
    ],
)
