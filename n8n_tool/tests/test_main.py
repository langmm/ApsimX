import copy
import pytest
import requests
import datetime


@pytest.fixture(scope="session")
def address():
    return "http://0.0.0.0:5000"


@pytest.fixture(scope="session")
def base_model_request():
    return {
        "crop_name": "Wheat",
        "crop_variety": "Hartog",
        "latitude": 40.1164,
        "longitude": -88.2434,
        "year": 1991,
        "start_date": "1991-01-01T00:00:00",
        "end_date": "1991-11-05T00:00:00",
        "output_vars": [
            "[Wheat].Grain.Total.Wt",
        ],
    }


def test_model(address, base_model_request):
    expected = pytest.approx(404.70110106813866, rel=1e-3)
    r = requests.post(f'{address}/start', json=base_model_request)
    r.raise_for_status()
    response = r.json()
    assert max(response['[Wheat].Grain.Total.Wt']) == expected


class TestInteractiveModel:

    @pytest.fixture(scope="class")
    @classmethod
    def model_request(cls, base_model_request):
        request = copy.deepcopy(base_model_request)
        request['action_step'] = 10
        request['actions'] = ["nitrogen"]
        return request

    @pytest.fixture(scope="class")
    @classmethod
    def idstr(cls, address, model_request):
        r = requests.post(
            f'{address}/start-interactive',
            json=model_request
        )
        r.raise_for_status()
        idstr = r.json()
        assert isinstance(idstr, str)
        try:
            yield idstr
            r = requests.post(f'{address}/{idstr}/stop')
            r.raise_for_status()
        except BaseException:
            r = requests.post(f'{address}/stop-interactive')
            r.raise_for_status()

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
            json={"names": list(value0.keys())}
        )
        r.raise_for_status()
        assert r.json() == value0
        # Set
        r = requests.put(
            interactive_address,
            json={"values": value1}
        )
        r.raise_for_status()
        assert r.json() == {"status": "success"}
        # Get
        r = requests.get(
            interactive_address,
            json={"names": list(value0.keys())}
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

    def test_act(self, interactive_address):
        r"""Test performing an action."""
        r = requests.post(
            f'{interactive_address}/act',
            json={
                "action_name": "nitrogen",
                "action_param": {
                    "amount": 160.0,  # kg/ha
                },
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
