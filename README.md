# easy_sdm
==============================

Species distribuition modelling for highly used plants in agricultural systems

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


### Atualizar repositório para esse formato 
https://github.com/crmne/cookiecutter-modern-datascience

### Rodando container

```bash
sudo docker build -t easy_sdm -f dockerfile .
sudo docker run -it --rm -p 8080:8080  --name easy_sdm -v $(pwd):/app easy_sdm
```

Entrar em um container que já está rodando. Pode acontecer quando se perde a conexão

```bash
sudo docker exec -it easy_sdm bash
```
### Subindo a FAST API

```bash
uvicorn api:app --port=8080 --host="0.0.0.0" --reload
```
### Fazendo chamada 
No diretório root chamar o arquivo a rodar como modulo da seguinte forma

```bash
python3 -m vident.document_reader.inference
```

### Testando as APIs
Se estiver dentro do container, tem que expor a porta 8080 

API de inferência
```bash
sudo curl -X POST -d '{"vector_type":"english_dense", "question":"Qual o contexto da palha de trigo?" , "ntops_retriever": 10, "ntops_overall": 5, "reader_score_weight": 0.8,"retriever_score_weight": 0.2 }'  http://0.0.0.0:8080/infer
```

API de update do backend completo
```bash
sudo curl -X POST -d '{"apply_filter":False, "apply_low_case":False, "num_gen_sentences": 10}'  http://0.0.0.0:8080/update_backend

```

### Visualizando performance com o tensorboard

tensorboard --logdir=~/Modelos/PLATIA/fabc_nlp/fabc_vident/logs/document_reader/lightning_logs/version_2










--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


