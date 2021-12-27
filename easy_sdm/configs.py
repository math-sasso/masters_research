import yaml
from pathlib import Path


configs = yaml.safe_load(open(Path(__file__).parent / "params.yaml", "r"))
