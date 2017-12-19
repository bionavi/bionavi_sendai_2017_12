# bionavi_sendai_2017_12

To setup the environment with required packages, grab a copy of anaconda from https://conda.io/miniconda.html and install it, in case not having it installed on your machine. After installation, activate root environment:

Unix:

source \path\to\anaconda\bin\activate

Windows

\path\to\anaconda\scripts\activate

Create a new enviroment "bionavi" using following command:

conda create -n bionavi python=3 scikit-learn pandas mkl mkl-service datashader dask pymc3

Then activate it by: Unix:

source \path\to\anaconda\bin\activate bionavi

Windows

\path\to\anaconda\scripts\activate bionavi

then install remaining packages using commands below:

conda install -c conda-forge -c ioam seaborn geopandas holoviews geoviews pip install graphviz

finally run jupyter notebook

jupyter notebook --notebook-dir=/path/to/notebook/folder
