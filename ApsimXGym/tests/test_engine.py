import numpy as np
import datetime
import logging
import contextlib
import pytest
from apsimx_gym import logger
from apsimx_gym.base import (
    InvalidActionError, RecoverableError,
    RecoverableModelEngineError,
)
from apsimx_gym.engine import ApsimXEngine, ApsimXEnv
logger.setLevel(logging.INFO)


class TestApsimXEngine:

    @pytest.fixture(scope="class")
    @classmethod
    def default_instance_kwargs(cls, apsimx_dir):
        return {
            "crop_name": "Wheat",
            "actions": ["nitrogen", "irrigate"],
            "model_dir": apsimx_dir,
            "start_time": datetime.datetime(year=1981, month=1, day=1),
            "end_time": datetime.datetime(year=1981, month=11, day=5),
        }

    @pytest.fixture(scope="class")
    @classmethod
    def new_instance(cls, example_model, default_instance_kwargs):

        @contextlib.contextmanager
        def _new_instance(model_file=None, **kwargs):
            if (("model_file" not in kwargs
                 and "crop_name" not in kwargs)):
                kwargs["model_file"] = example_model
            for k, v in default_instance_kwargs.items():
                kwargs.setdefault(k, v)
            instance = ApsimXEngine(**kwargs)
            instance.start()
            try:
                yield instance
            finally:
                instance.stop()
                instance.model.cleanup()

        return _new_instance

    @pytest.fixture(scope="class")
    @classmethod
    def instance(cls, new_instance):
        with new_instance(model_suffix="-Prime") as instance:
            yield instance

    def test_attributes(self, instance):
        r"""Test instance attributes."""
        assert instance.crop_name == "Wheat"
        assert instance.crop_variety == "Hartog"
        assert instance.location != "the field"
        assert instance.field_area == 1.0
        print(instance.get_output_vars())

    @pytest.mark.parametrize(
        "command,error,command_args", [
            ("set", RecoverableModelEngineError, (
                "invalid", 1,
            )),
            ("get", RecoverableModelEngineError, (
                "invalid",
            )),
            ("act", InvalidActionError, (
                "invalid",
            )),
            ("set", RecoverableModelEngineError, (
                "[Clock].Start",
                datetime.datetime(year=1981, month=1, day=1).timestamp()
            )),
            ("act", RecoverableModelEngineError, (
                "nitrogen", "hello",
            )),
        ], ids=[
            "set", "get", "act", "set_value", "act_value",
        ]
    )
    def test_engine_invalid(self, instance, new_instance,
                            command, error, command_args):
        r"""Test error & cleanup on setting a variable that causes
        re-initialization."""
        with new_instance() as instance2:
            with pytest.raises(error):
                getattr(instance2, command)(*command_args)
            if issubclass(error, RecoverableError):
                assert instance2.is_running
                assert not instance2.is_complete
                assert instance2.status == "paused"
            else:
                assert not instance2.is_running
                assert not instance2.is_complete
                assert instance2.status == "error"
        if issubclass(error, RecoverableError):
            getattr(instance, command)(*command_args, allow_error=True)
            assert instance.is_running
            assert not instance.is_complete
            assert instance.status == "paused"

    def test_loop(self, new_instance):
        r"""Test loop to apply fertilizer and irrigate."""
        with new_instance(
                latitude=40.1164, longitude=-88.2434,
                start_time=datetime.datetime(year=1984, month=7, day=1),
                end_time=datetime.datetime(year=1985, month=5, day=5),
        ) as instance:
            i = 0
            orig_value = 0.05
            new_value = 0.043
            reply = instance.get(
                "[Grain].MaximumPotentialGrainSize.FixedValue")
            assert reply == orig_value
            instance.set(
                "[Grain].MaximumPotentialGrainSize.FixedValue",
                new_value)
            reply = instance.get(
                "[Grain].MaximumPotentialGrainSize.FixedValue")
            assert reply == new_value
            while instance.is_running and not instance.is_complete:
                instance.getvars([
                    "[Wheat].Phenology.Zadok.Stage",
                    "[Soil].Water.PAW",
                ])
                # Decision point
                if i % 2 == 0:
                    instance.act("nitrogen", 160.0)  # kg/ha
                else:
                    instance.act("irrigate", 10.0)  # mm
                # reply = instance.get("[Nutrient].NO3.kgha")
                # new_value = [2*ele for ele in reply]
                # instance.set("[Nutrient].NO3.kgha", new_value)
                # reply = instance.get("[Nutrient].NO3.kgha")
                # assert reply == new_value
                reply = instance.getvars([
                    "[Wheat].Phenology.Zadok.Stage",
                    "[Wheat].Grain.Total.Wt",
                    "[Fertiliser].NitrogenApplied",
                    "[Irrigation].IrrigationApplied",
                ])
                reply["[Clock].Today"] = instance.current_time
                assert isinstance(reply, dict)
                # import pprint
                # pprint.pprint(reply)
                instance.fast_forward(datetime.timedelta(days=10))
                i += 1

    def test_scrub(self, instance):
        r"""Test rewind/fast-forward."""
        start_time = instance.start_time
        end_time = instance.end_time
        # start_time = datetime.datetime(year=1981, month=1, day=1)
        # end_time = datetime.datetime(year=1981, month=11, day=5)
        time = start_time.replace(month=5, day=23)
        # Run complete simulation without fertilizing
        assert instance.current_time == start_time
        instance.fast_forward()
        assert instance.current_time == end_time
        assert instance.is_running
        value_none = instance.get("[Wheat].Grain.Total.Wt")
        # print("NONE", instance.current_time, value_none)
        # Rewind and run again with full
        instance.rewind()
        instance.rewind(datetime.timedelta(days=20))
        assert instance.current_time == start_time
        while instance.is_running and not instance.is_complete:
            instance.act("nitrogen", 0.001)
            instance.fast_forward(datetime.timedelta(days=20))
        assert instance.current_time == end_time
        value_full = instance.get("[Wheat].Grain.Total.Wt")
        # print("FULL", instance.current_time, value_full)
        # Rewind halfway and run without from there
        instance.rewind(time)
        assert instance.current_time == time
        instance.fast_forward()
        assert instance.current_time == end_time
        instance.fast_forward(datetime.timedelta(days=20))
        value_half = instance.get("[Wheat].Grain.Total.Wt")
        # print("HALF", instance.current_time, value_half)
        # Compare
        assert value_half > value_none
        assert value_full > value_half
        instance.rewind()
        assert instance.current_time == start_time

    def test_action_param(self, new_instance):
        r"""Test actions with parameters."""
        with new_instance(
                actions=[
                    "nitrogen", "irrigate", "tillage", "sow", "harvest"
                ],
                action_param={
                    "sow": {"population": 10.0, "crop_name": "Wheat"},
                },
        ) as instance:
            instance.act("tillage", "disc")
            instance.act("nitrogen", 160.0, type="UreaN")
            instance.act("sow", sowingDepth=10.0)
            instance.fast_forward(
                datetime.datetime(year=1981, month=5, day=23))
            instance.act("harvest")


class TestApsimXEnv:
    r"""Test use of ApsimXEnv for exclusive, discrete actions."""

    @pytest.fixture(scope="class")
    @classmethod
    def default_instance_kwargs(cls, apsimx_dir):
        return {
            "model_dir": apsimx_dir,
            "start_time": datetime.datetime(year=1981, month=1, day=1),
            "end_time": datetime.datetime(year=1981, month=12, day=31),
        }

    @pytest.fixture(scope="class")
    @classmethod
    def new_instance(cls, example_model, default_instance_kwargs):

        @contextlib.contextmanager
        def _new_instance(model_file=None, **kwargs):
            if (("model_file" not in kwargs
                 and "crop_name" not in kwargs)):
                kwargs["model_file"] = example_model
            for k, v in default_instance_kwargs.items():
                kwargs.setdefault(k, v)
            env = ApsimXEnv(**kwargs)
            try:
                yield env
            finally:
                env.close()
                env.model.model.cleanup()

        return _new_instance

    @pytest.fixture(scope="class")
    @classmethod
    def instance(cls, new_instance):
        with new_instance(model_suffix="-Prime") as instance:
            yield instance

    @pytest.fixture(scope="class")
    @classmethod
    def sampled_action_id(cls, instance):
        return instance.action_map.space.sample()

    @pytest.fixture(params=[0, 1])
    @classmethod
    def action_id(cls, request):
        r"""int: Action ID."""
        return request.param

    @pytest.fixture(params=[
        {
            "nitrogen": np.array([0.5]),
            "irrigate": np.array([0.5]),
        },
        (1, np.array([0.5])),
    ])
    @classmethod
    def invalid_action_id(cls, request):
        r"""int: Action ID."""
        return request.param

    def test_description(self, instance):
        r"""Test description generation."""
        instance.action_map.description

    def test_action_description(self, instance, action_id,
                                sampled_action_id,
                                assert_nested_allclose):
        r"""Test description generation."""
        desc = instance.action_map.action2description(action_id)
        assert instance.action_map.description2action(desc) == action_id
        desc = instance.action_map.action2description(sampled_action_id)
        assert_nested_allclose(
            instance.action_map.description2action(desc),
            sampled_action_id,
            atol=0.1
        )

    def test_invalid_action(self, instance, invalid_action_id):
        with pytest.raises(InvalidActionError):
            instance.action_map.action2description(invalid_action_id)

    def test_step(self, instance, sampled_action_id):
        r"""Test environment step."""
        instance.step(sampled_action_id)
        instance.reset()

    def test_prompt_generator(self, instance, action_id,
                              sampled_action_id, assert_nested_allclose):
        r"""Test creation of prompt generator."""
        prompt = instance.get_llm_prompt_generator()
        prompt.get_system_prompt()
        desc = prompt.describe_action(action_id)
        assert prompt.parse_action_response(desc) == action_id
        desc = prompt.describe_action(sampled_action_id)
        assert_nested_allclose(
            prompt.parse_action_response(desc), sampled_action_id,
            atol=0.1
        )
        assert prompt.parse_action_response("Invalid response") is None


class TestApsimXEnvContinuous(TestApsimXEnv):
    r"""Test use of ApsimXEnv with exclusive, continuous actions."""

    @pytest.fixture(scope="class")
    @classmethod
    def default_instance_kwargs(cls, apsimx_dir):
        return {
            "model_dir": apsimx_dir,
            "start_time": datetime.datetime(year=1981, month=1, day=1),
            "end_time": datetime.datetime(year=1981, month=12, day=31),
            "num_levels": 0,
        }

    @pytest.fixture(params=[
        (0, 0),
        (1, np.array([0.5])),
    ])
    def action_id(self, request):
        r"""int: Action ID."""
        return request.param

    @pytest.fixture(params=[
        0,
        {
            "nitrogen": np.array([0.5]),
            "irrigate": np.array([0.5]),
        },
    ])
    def invalid_action_id(self, request):
        r"""int: Action ID."""
        return request.param


class TestApsimXEnvSimultaneous(TestApsimXEnv):
    r"""Test use of ApsimXEnv with non-exclusive, continuous actions."""

    @pytest.fixture(scope="class")
    @classmethod
    def default_instance_kwargs(cls, apsimx_dir):
        return {
            "model_dir": apsimx_dir,
            "start_time": datetime.datetime(year=1981, month=1, day=1),
            "end_time": datetime.datetime(year=1981, month=12, day=31),
            "exclusive": False,
            "num_levels": 0,
        }

    @pytest.fixture(params=[
        {
            "nitrogen": np.array([0.5]),
            "irrigate": np.array([0.5]),
        },
    ])
    def action_id(self, request):
        r"""int: Action ID."""
        return request.param

    @pytest.fixture(params=[
        0,
        (1, np.array([0.5])),
        np.zeros((5, )),
    ])
    def invalid_action_id(self, request):
        r"""int: Action ID."""
        return request.param
