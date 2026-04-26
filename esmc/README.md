Instructions:

```console
$ docker build -t esmc .
$ docker run -it --rm --gpus '"device=0"' -v ./.huggingface:/root/.cache/huggingface esmc bash
# in container:
$ python run_esmc.py
```
