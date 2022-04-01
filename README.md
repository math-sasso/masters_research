# easy_sdm
==============================

Species distribuition modelling for highly used plants in agricultural systems

### Building container

```bash
sudo docker build -t easy_sdm -f dockerfile .
sudo docker run -it --rm -p 8080:8080  --name easy_sdm -v $(pwd):/app easy_sdm
sudo docker run -it --rm --name easy_sdm -v $(pwd):/app easy_sdm
```
###  Entering in a running container
```bash
sudo docker exec -it easy_sdm bash
```

## Studied species:

MILPA

```bash
    "Zea mysa": 5290052,
    "Cucurbita moschata": 7393329,
    "Cucurbita maxima": 2874515,
    "Cucurbita pepo": 2874508,
    "Phaseolus vulgaris": 5350452,
    "Vigna unguiculata": 2982583,
    "Cajanus cajan": 7587087,
    "Piper nigrum": 3086357,
    "Capsicum annuum": 2932944,
    "Capsicum baccatum": 2932938,
    "Capsicum frutescens": 8403992,
    "Capsicum chinense": 2932942,
```

Milpa Rare

```bash
    "Zea diploperennis": 5290048,
    "Zea luxurians": 5290053,
    "Zea nicaraguensis": 5678409,
    "Zea perennis": 5290054,
    "Cucurbita ficifolia": 2874512,
    "Cucurbita argyrosperma": 7907172,
    "Capsicum pubescens": 2932943,
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
