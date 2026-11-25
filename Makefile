PYTEST = uv run --with pytest python -m pytest -q
PLE = tests/test_assignment.py
KUMO = tests/test_kumo.py

.PHONY: test check test_data test_ple_encoding test_ple_init test_ple_pack test_ple_speed \
        test_kumo_cells test_kumo_encoder test_kumo_icl test_kumo_weights benchmark

test check:
	$(PYTEST) tests

test_data:
	$(PYTEST) $(PLE)::test_train_quantiles

test_ple_encoding:
	$(PYTEST) $(PLE) -k 'coefficients or piecewise_linear_encoding or reference_width'

test_ple_init:
	$(PYTEST) $(PLE)::test_embedding_init

test_ple_pack:
	$(PYTEST) $(PLE)

test_ple_speed:
	$(PYTEST) -s tests/test_ple_speed.py

test_kumo_cells:
	$(PYTEST) $(KUMO) -k 'group or cells'

test_kumo_encoder:
	$(PYTEST) '$(KUMO)::test_stages[encoder]'

test_kumo_icl:
	$(PYTEST) '$(KUMO)::test_stages[icl]'

test_kumo_weights:
	KUMO_REG=weights/regressor.pt KUMO_CLS=weights/classifier.pt $(PYTEST) $(KUMO) -k released_checkpoint

benchmark:
	uv run python -m lib.run
