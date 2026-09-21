.PHONY: all verify test fixture clean

all: verify test

verify:  ## Reproduce every arithmetic finding in the audit
	python3 tools/verify_manuscript_arithmetic.py --out reports/

test:  ## Run the regression suite
	python3 -m pytest tests/ -q

fixture:  ## Regenerate the synthetic test fixture (testing only)
	python3 tools/make_synthetic_fixture.py --out tests/fixtures

clean:
	rm -rf reports/*.csv reports/*.json .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
