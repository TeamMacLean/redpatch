from setuptools import setup

setup(
    name='redpatch',
    version='0.0.1dev8',
    packages=['redpatch', 'redpatch_notebooks'],
    url='https://github.com/TeamMacLean/redspot',
    license='LICENSE.txt',
    author='Dan MacLean',
    author_email='dan.maclean@tsl.ac.uk',
    description='Finding Disease Lesions in Plant Leaves',
    scripts=['scripts/redpatch-start'],
    include_package_data=True,
    package_data={"redpatch_notebooks": ['Redpatch Basic Use Example.ipynb']},
    python_requires='>=3.7',
    install_requires=[
        "ipywidgets == 7.5.1",
        "jupyter == 1.0.0",
        "numpy == 1.17.1",
        "matplotlib >= 3.1.0",
        "pytest == 5.1.2",
        "scikit-image >= 0.15.0",
        "scipy == 1.3.1"
    ],
)
