
# Project

This project is an experiment for the report, aimed at testing the convergence of the algorithm Cluster-Based Secure Multi-Party Computation Decentralized Learning (CSMPCDL) proposed in the report. A curve plot will be generated in this document to compare the convergence performance of CSMPCDL, decentralized learning, and federated learning.

## Requirements


To install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Data Generation

Run the `data_generation.py` script to generate the dataset. This will create two files:

- `data.csv`: The training set with 16,000 samples.
- `test.csv`: The test set.

To generate the data, use the following command:

```bash
python data_generation.py
```

After running the script, you will find the `data.csv` and `test.csv` files in the project directory.

## Running Training and Testing

Run the `main.py` script to train using five methods, each for 100 iterations, and calculate the average results. The script will automatically generate a plot comparing the methods after running.

To run the script, use the following command:

```bash
python main.py
```

After running, you will see an automatically generated plot displaying the mean results for each method.