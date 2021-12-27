# easy_sdm
==============================

Species distribuition modelling for highly used plants in agricultural systems

### Building container

```bash
sudo docker build -t easy_sdm -f dockerfile .
sudo docker run -it --rm -p 8080:8080  --name easy_sdm -v $(pwd):/app easy_sdm
```
###  Entering in a running container
```bash
sudo docker exec -it easy_sdm bash
```



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


