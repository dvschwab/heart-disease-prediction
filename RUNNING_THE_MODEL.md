# Running the model locally

This assumes that you are running the model in Jupyter Lab. The process for another IDE should be similar.

First, create a new environment using conda based on the environment specification in the YAML file:

> `conda env create --file heart-disease-deploy-env.yaml`

This will download and unpack all the required packages for the notebook to run. Switch to that environment:

> `conda activate heart-disease-deploy`

This sets up the environment, but Jupyter may not be able to find it without this next step. If that is the case, then do the following:

1. Using conda, install the package `ipykernel`:

> `conda install -c anaconda ipykernel`

2. Run the following to make a new Jupyter kernel based on the heart-disease-deploy environment

> `python -m ipykernel install --user --name=heart-disease-deploy`

3. Open Jupyter (Lab or Notebook) and select **Kernel | Change Kernel**. Select the `heart-disease-deploy` kernel.

If all goes well, Jupyter should now use the `heart-disease-deploy` environment containing all the packages necessary to run the notebook.