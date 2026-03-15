.PHONY: install test

install:
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

test:
python -m pytest
