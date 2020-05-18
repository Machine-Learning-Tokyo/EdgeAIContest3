# EdgeAIContest3
This repository is to work on the [The 3rd AI Edge Contest](https://signate.jp/competitions/256).

## Structure
    .
    ├── src             # Source files (pre/post/processing)
    ├── test            # Testing files (small unit tests etc.)
    ├── model           # Models (binary/json model files)
    ├── data            # Data (augmented/raw/processed)
    ├── notebook        # Jupyter Notebooks (exploration/modelling/evaluation)
    ├── LICENSE         
    └── README.md       

## Setup data
Please download data from the competition into the ```data/``` folder:
*After setting the signate CLI [link](https://pypi.org/project/signate/)*
```
cd data/
signate download --competition-id=256
```