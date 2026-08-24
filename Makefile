JAVA_HOME ?= /usr/lib/jvm/java-17-amazon-corretto
GRADLE := JAVA_HOME=$(JAVA_HOME) ./gradlew

.PHONY: help build test retest coverage bench demo format check-format publish-local publish clean install-hooks

help:
	@echo "Targets:"
	@echo "  build          Compile main + test sources"
	@echo "  test           Run the JUnit test suite"
	@echo "  retest         Re-run the test suite, ignoring Gradle's up-to-date cache"
	@echo "  coverage       Run tests, generate the JaCoCo report, and print a summary"
	@echo "  bench          Run the JMH prediction benchmark"
	@echo "  demo           Run the end-to-end Iris train-then-predict tutorial"
	@echo "  format         Apply code formatting and import sorting (Spotless)"
	@echo "  check-format   Verify formatting without modifying files"
	@echo "  publish-local  Publish jars to the local ~/.m2 repository"
	@echo "  publish        Publish signed jars to Maven Central (needs GPG + Central Portal env)"
	@echo "  install-hooks  Install the git pre-commit hook"
	@echo "  clean          Remove build outputs"

build:
	$(GRADLE) build -x test

test:
	$(GRADLE) test

retest:
	$(GRADLE) test --rerun-tasks

coverage:
	$(GRADLE) test jacocoTestReport
	@python3 scripts/coverage_summary.py

bench:
	$(GRADLE) jmh

demo:
	@$(GRADLE) demo -q --console=plain

format:
	$(GRADLE) spotlessApply

check-format:
	$(GRADLE) spotlessCheck

install-hooks:
	git config core.hooksPath .githooks

publish-local:
	$(GRADLE) publishToMavenLocal

publish:
	$(GRADLE) publishAndReleaseToMavenCentral

clean:
	$(GRADLE) clean
