PY := .venv/bin/python
SEEDS := 0 1 2

.PHONY: setup data train eval figures test lint demo ablations all clean

setup:
	test -d .venv || uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.lock

data:
	$(PY) scripts/download_data.py

train:
	for s in $(SEEDS); do \
		$(PY) -m molfp.train --config configs/esol_gcn.yaml --seed $$s || exit 1; \
		$(PY) -m molfp.train --config configs/tox21_gcn.yaml --seed $$s || exit 1; \
	done

eval:
	$(PY) -m molfp.evaluate --dataset esol
	$(PY) -m molfp.evaluate --dataset tox21

figures:
	$(PY) -m molfp.figures

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

demo:
	$(PY) -m streamlit run app/streamlit_app.py

ablations:
	$(PY) -m molfp.evaluate --ablations

all:
	@start=$$(date +%s); \
	$(MAKE) setup data train eval figures test || exit 1; \
	end=$$(date +%s); echo "make all finished in $$(( (end-start)/60 )) min $$(( (end-start)%60 )) s"

clean:
	rm -rf runs .pytest_cache .ruff_cache
