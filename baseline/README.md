# Baseline Model
Our implementation of a baseline model against which to compare our results.

## Usage Instructions
1. Install Packages: Follow the instructions in the provided `requirements.txt` file
   to install the necessary packages for running our code.

   Then, download the necessary NLTK libraries by running our provided script:

   `python nltk-setup.py`

2. Providing Data: Add the dataset to train and evaluation the model on in the
   `/data` directory under a subfolder name of your choice. Within
   the dataset's subfolder, each file should use the format:

   `[corpus|annotation].[train|val|test].xml`

   The dataset `tiny` is provided as an example. Once the dataset is placed
   correctly, modify `configuration.py` to point to the dataset in the
   `DATA` configuration constant.

3. Producing Summaries: Run the following commands to generate the
   reference and system summaries (respectively) from the provided dataset:

   `python NB_bc3_reference.py`

   `python NB_bc3.py`

4. Evaluation: Run the following command to use the ROUGE system for evaluation:

   `java -jar rouge2-1.2.1.jar`
