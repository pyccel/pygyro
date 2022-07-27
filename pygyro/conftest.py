
def pytest_addoption(parser):
    parser.addoption("--short", action="store_true",
                     help="run fewer tests (to be used when running in pure python mode)")
