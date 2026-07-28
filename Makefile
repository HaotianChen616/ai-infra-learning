.PHONY: test lab-kv lab-scheduler lab-prefix lab-h2d-d2h lab-h2d-d2h-ascend

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

lab-h2d-d2h:
	$(PYTHON) labs/h2d_d2h_benchmark.py \
		--backend cuda \
		--sizes 4KiB,1MiB,16MiB,64MiB \
		--output-dir artifacts/h2d_d2h

lab-h2d-d2h-ascend:
	$(PYTHON) labs/h2d_d2h_benchmark.py \
		--backend npu \
		--sizes 4KiB,1MiB,16MiB,64MiB \
		--output-dir artifacts/ascend_h2d_d2h
