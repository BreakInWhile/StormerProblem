test_poincare:
	python3 -m unittest tests/unit/test_poincare.py

test_methods:
	python3 -m unittest tests.unit.test_methods

test_c:
	python3 -m unittest tests/unit/test_c.py

experiments:
	python3 -m tests

ccompile:
	cd tfg/methods/optim && python3 setup.py build_ext --inplace
