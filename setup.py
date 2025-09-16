from setuptools import setup, find_packages

setup(
    name="network-intelligence-system",
    version="1.0.0",
    description="AI-powered SDN network monitoring and management system",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "mininet>=2.3.0",
        "ryu>=4.34", 
        "scapy>=2.4.5",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "flask>=2.0.0",
        "requests>=2.28.0",
        "psutil>=5.8.0",
        "networkx>=2.6.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)