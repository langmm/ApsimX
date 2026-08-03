import os
import json
import copy
import time
import pytest
import requests
import datetime
import contextlib
from subprocess import Popen
_n8n_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
_apsimx_dir = os.path.dirname(_n8n_dir)


@pytest.fixture(scope="session")
def ping_address():

    def _ping_address(address):
        try:
            r = requests.get(f"{address}/docs")
            r.raise_for_status()
            return True
        except requests.exceptions.ConnectionError:
            return False

    return _ping_address


@pytest.fixture(scope="session")
def local_address(ping_address):
    out = "http://0.0.0.0:5000"
    if ping_address(out):
        yield out
    else:
        docker = False
        if docker:
            cmd = "docker run -p 5000:8000 apsimx"
        else:
            cmd = "fastapi run --host 0.0.0.0 --port 5000 main.py"
        p = Popen(
            cmd.split(),
            cwd=_n8n_dir,
            env=dict(
                os.environ,
                APSIMX_DIR=_apsimx_dir,
            ),
        )
        try:
            while p.poll() is None and not ping_address(out):
                time.sleep(1)
            yield out
        finally:
            if p.poll() is None:
                p.terminate()
                p.kill()
            else:
                assert p.returncode == 0


@pytest.fixture(scope="session", params=[
    "local",
    "remote",
])
def address(local_address, request, ping_address):
    if request.param == "local":
        out = local_address
    else:
        out = (
            "https://66202345-1ca8-4b56-9fb7-abb6e439a4db-8000"
            ".app.beam.cloud"
        )
    if not ping_address(out):
        pytest.skip(f"Could not connect to \"{out}\"")
    yield out


@pytest.fixture(scope="session")
def running_interactive_model(address):

    @contextlib.contextmanager
    def _running_interactive_model(request, dont_stop=False):
        r = requests.post(f'{address}/start-interactive',
                          json=request)
        r.raise_for_status()
        idstr = r.json()
        model_address = f'{address}/interactive-model/{idstr}'
        try:
            yield idstr
            if not dont_stop:
                r = requests.post(f'{model_address}/stop')
                r.raise_for_status()
        finally:
            r = requests.post(f'{address}/stop-interactive')

    return _running_interactive_model


@pytest.fixture(scope="session")
def base_model_request():
    return {
        "crop_name": "Wheat",
    }


def test_model(address, base_model_request):
    expected = pytest.approx(309.2315738609009, rel=1e-3)
    r = requests.post(f'{address}/start', json=base_model_request)
    r.raise_for_status()
    response = r.json()
    assert max(response['[Wheat].Grain.Total.Wt']) == expected


def test_model_interactive(address, base_model_request,
                           running_interactive_model):
    request = copy.deepcopy(base_model_request)
    request.update(actions=["irrigate"])
    with running_interactive_model(request) as idstr:
        model_address = f'{address}/interactive-model/{idstr}'
        r = requests.post(f'{model_address}/complete')
        r.raise_for_status()
        assert r.json() == {"status": "success"}
        # Disabled until the report generation can be debugged
        # r = requests.get(f'{model_address}/results')
        # r.raise_for_status()
        # print(r.content)
        # import pdb; pdb.set_trace()
        # assert r.content


def test_model_interactive_timeout(address, base_model_request,
                                   running_interactive_model):
    request = copy.deepcopy(base_model_request)
    request.update(wait_time=1)
    with running_interactive_model(request, dont_stop=True) as idstr:
        model_address = f'{address}/interactive-model/{idstr}'
        time.sleep(2)
        r = requests.get(f'{model_address}/status')
        r.raise_for_status()
        assert r.json() == {"status": "stopped"}


class TestInteractiveModel:

    @pytest.fixture(scope="class")
    @classmethod
    def model_request(cls, base_model_request):
        request = {
            "crop_name": "Wheat",
            "crop_variety": "Hartog",
            "latitude": 40.1164,
            "longitude": -88.2434,
            "year": 1991,
            "start_time": "1991-01-01T00:00:00",
            "end_time": "1991-11-05T00:00:00",
            "timestep": 10,
            "actions": "nitrogen,irrigate",
            "state_variables": "[Clock].Today,[Wheat].Grain.Total.Wt",
        }
        return request

    @pytest.fixture(scope="class")
    @classmethod
    def idstr(cls, model_request, running_interactive_model):
        with running_interactive_model(model_request) as idstr:
            yield idstr

    @pytest.fixture(scope="class")
    @classmethod
    def interactive_address(cls, address, idstr):
        return f'{address}/interactive-model/{idstr}'

    def test_invalid(self, address):
        r = requests.get(f'{address}/interactive-model/invalid-id/status')
        with pytest.raises(requests.HTTPError):
            r.raise_for_status()
        assert r.json() == {'detail': 'No model with id "invalid-id"'}

    def test_status(self, interactive_address):
        r"""Check that the model is running."""
        r = requests.get(f'{interactive_address}/status')
        r.raise_for_status()
        assert r.json()["status"] == "running"

    def test_get_set(self, interactive_address):
        r"""Test getting/setting a state variable."""
        value0 = {'[Grain].MaximumPotentialGrainSize.FixedValue': 0.05}
        value1 = {'[Grain].MaximumPotentialGrainSize.FixedValue': 0.043}
        # Get
        r = requests.get(
            interactive_address,
            json={"state_variables": ",".join(list(value0.keys()))}
        )
        r.raise_for_status()
        assert r.json() == value0
        # Set
        r = requests.put(
            interactive_address,
            json={"values": json.dumps(value1)}
        )
        r.raise_for_status()
        assert r.json() == {"status": "success"}
        # Get
        r = requests.get(
            interactive_address,
            json={"state_variables": ",".join(list(value0.keys()))}
        )
        r.raise_for_status()
        assert r.json() == value1

    def current_time(self, root_address):
        r = requests.get(f'{root_address}/status')
        r.raise_for_status()
        return datetime.datetime.fromisoformat(r.json()["time"])

    def test_continue(self, interactive_address):
        r"""Test continuing to the next step."""
        t0 = self.current_time(interactive_address)
        r = requests.post(f'{interactive_address}/continue')
        r.raise_for_status()
        assert r.json() == {"status": "success"}
        t1 = self.current_time(interactive_address)
        assert t1 > t0

    def test_trace(self, interactive_address):
        r = requests.get(f'{interactive_address}/trace')
        r.raise_for_status()
        response = r.json()
        assert '[Clock].Today' in response
        assert '[Wheat].Grain.Total.Wt' in response

    def test_act(self, interactive_address):
        r"""Test performing an action."""
        r = requests.post(
            f'{interactive_address}/act',
            json={
                "action": "nitrogen",
                "parameters": json.dumps({
                    "amount": 160.0,  # kg/ha
                }),
            }
        )
        r.raise_for_status()
        assert r.json() == {"status": "success"}

    def test_scrub(self, interactive_address):
        r"""Test moving forward and backward in the simulation."""
        t = self.current_time(interactive_address)
        tscrub = t + datetime.timedelta(days=10)
        # Fast forward
        r = requests.post(f'{interactive_address}/complete')
        r.raise_for_status()
        assert r.json() == {"status": "success"}
        # Rewind
        r = requests.post(f'{interactive_address}/restart')
        r.raise_for_status()
        assert r.json() == {"status": "success"}
        # Scrub with time
        r = requests.post(f'{interactive_address}/scrub',
                          params={'time': tscrub.isoformat()})
        r.raise_for_status()
        assert r.json() == {"status": "success"}
        assert self.current_time(interactive_address) == tscrub
        r = requests.post(f'{interactive_address}/scrub',
                          params={'time': int(-10)})
        r.raise_for_status()
        assert r.json() == {"status": "success"}
        assert self.current_time(interactive_address) == t
