.PHONY: test lab-kv lab-scheduler lab-prefix

PYTHON ?= python3

test:
	$(PYTHON) -m unittest discover -s tests -v

lab-kv:
	$(PYTHON) labs/kv_cache_calculator.py \
		--layers 32 --kv-heads 8 --head-dim 128 --dtype bf16 \
		--context-length 8192 --concurrency 32 --tp-size 4 \
		--kv-capacity-gib-per-device 20

lab-scheduler:
	$(PYTHON) labs/scheduler_simulator.py --show-trace

lab-prefix:
	$(PYTHON) labs/prefix_cache_simulator.py
