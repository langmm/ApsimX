import os
import numpy as np
import pytest


@pytest.fixture(scope="session")
def apsimx_dir():
    r"""str: Path to the apsimx directory."""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="session")
def example_model(apsimx_dir):
    r"""str: Path to example model input file."""
    return os.path.join(apsimx_dir, "Examples", "Wheat.apsimx")


class NestedAssertionError(AssertionError):

    def __init__(self, nested):
        self.nested = nested
        msg = ''
        for k, v in nested.items():
            msg += f'\n\n{k}\n\t' + v.replace('\n', '\n\t')
        super(NestedAssertionError, self).__init__(msg)


@pytest.fixture(scope="session")
def assert_allclose():
    r"""Assert that arrays are close."""

    def _assert_allclose(a, b, rtol=1e-07, atol=1e-15, **kwargs):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, **kwargs)

    return _assert_allclose


@pytest.fixture(scope="session")
def assert_nested_allclose(assert_allclose):
    from collections import OrderedDict

    def _assert_nested_allclose(a, b, ignore_keys=None,
                                only_keys=None, **kwargs):
        errors = {}
        if isinstance(b, (list, tuple)):
            assert isinstance(a, type(b))
            assert len(a) == len(b)
            for i, (ia, ib) in enumerate(zip(a, b)):
                try:
                    _assert_nested_allclose(ia, ib, **kwargs)
                except AssertionError as e:
                    if isinstance(e, NestedAssertionError):
                        for kerr, verr in e.nested.items():
                            errors[f'{i}->{kerr}'] = verr
                    else:
                        errors[f'{i}'] = e.args[0]
        elif isinstance(b, (dict, OrderedDict)):
            assert isinstance(a, type(b))
            a_keys = list(sorted(a.keys()))
            b_keys = list(sorted(b.keys()))
            if ignore_keys:
                a_keys = [k for k in a_keys if k not in ignore_keys]
                b_keys = [k for k in b_keys if k not in ignore_keys]
            if only_keys:
                a_keys = [k for k in a_keys if k in only_keys]
                b_keys = [k for k in b_keys if k in only_keys]
            assert a_keys == b_keys
            for k in b_keys:
                try:
                    _assert_nested_allclose(a[k], b[k], **kwargs)
                except AssertionError as e:
                    if isinstance(e, NestedAssertionError):
                        for kerr, verr in e.nested.items():
                            errors[f'{k}->{kerr}'] = verr
                    else:
                        errors[k] = e.args[0]
        elif isinstance(b, (np.ndarray, float)):
            assert_allclose(a, b, **kwargs)
        else:
            assert a == b
        if errors:
            raise NestedAssertionError(errors)

    return _assert_nested_allclose
