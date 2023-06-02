# chemlearn
Applied machine learning for computational chemistry.

![Workflow](https://github.com/marcossantanaioc/chemlearn/actions/workflows/pipeline.yml/badge.svg) [![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

# How to use
chemlearn handles all steps from featurization of molecules, training of machine learning/deep learning models
and validation. 

### MolDataset
The `MolDataset` class handles featurization and data splitting. The current version only supports pandas Dataframes,
and the input must be a string representing a filepath to a CSV file. The other inputs to `MolDataset` are
a splitting strategy, using one of the factory [splitters](chemlearn/utils/splitters.py), and a featurizer ([See cheminftools for a list of available featurizers](https://github.com/marcossantanaioc/cheminftools/blob/main/cheminftools/tools/featurizer.py))

```python
from chemlearn.data.dataset import MolDataset
from cheminftools.tools.featurizer import MolFeaturizer
from chemlearn.utils.splitters import TrainTestSplitter

# Input data
csv_path = 'example_data.csv'

# Column with SMILES
smiles_column = 'smiles'

# Prediction target
target_column = 'pIC50'

# Define splitting strategy
splitter = TrainTestSplitter(test_size=0.2)
# Define featurizer
featurizer = MolFeaturizer(descriptor_type='morgan', params={'radius':3, 'fpSize': 2048})


dataset = MolDataset(data_path=csv_path, smiles_column=smiles_column, target_variable=target_column, splitter=splitter, featurizer=featurizer)

```

# Minimal example: training a ChemLearner

```python
from chemlearn.models.skmodel import ChemLearner
from sklearn.ensemble import RandomForestRegressor
from chemlearn.models.validation.metrics import R2Score
from chemlearn.data.dataset import MolDataset
from cheminftools.tools.featurizer import MolFeaturizer
from chemlearn.utils.splitters import TrainTestSplitter

metric = R2Score()
# Input data
csv_path = 'example_data.csv'

# Column with SMILES
smiles_column = 'smiles'

# Prediction target
target_column = 'pIC50'

# Define splitting strategy
splitter = TrainTestSplitter(test_size=0.2)

# Define featurizer
featurizer = MolFeaturizer(descriptor_type='morgan', params={'radius':3, 'fpSize': 2048})

dataset = MolDataset(data_path=csv_path, smiles_column=smiles_column, target_variable=target_column, splitter=splitter, featurizer=featurizer)

chem_learner = ChemLearner(model=RandomForestRegressor(), dataset=dataset, metric=metric)
chem_learner.fit(dataset)
```