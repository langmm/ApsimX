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
    def new_instance(self, example_model, apsimx_dir):

        @contextlib.contextmanager
        def _new_instance(model_file=None, **kwargs):
            if model_file is None:
                model_file = example_model
            kwargs.setdefault("apsimx_dir", apsimx_dir)
            kwargs.setdefault("actions", ["nitrogen", "irrigate"])
            kwargs.setdefault(
                "start_time",
                datetime.datetime(year=1900, month=1, day=1))
            kwargs.setdefault(
                "end_time",
                datetime.datetime(year=1900, month=12, day=31))
            instance = ApsimXEngine(model_file, **kwargs)
            instance.start()
            try:
                yield instance
            finally:
                instance.stop()
                instance.model.cleanup()

        return _new_instance

    @pytest.fixture(scope="class")
    def instance(self, new_instance):
        with new_instance(model_suffix="-Prime") as instance:
            yield instance

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
                datetime.datetime(year=1900, month=1, day=1).timestamp()
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
        with new_instance() as instance:
            i = 0
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

    def test_scrub(self, new_instance):
        r"""Test rewind/fast-forward."""
        # TODO: Use existing instance
        with new_instance(
            start_time=datetime.datetime(year=1900, month=1, day=1),
            end_time=datetime.datetime(year=1900, month=11, day=5),
        ) as instance:
            # time = datetime.datetime(year=1900, month=9, day=22)
            time = datetime.datetime(year=1900, month=5, day=22)
            # Run complete simulation without fertilizing
            instance.fast_forward()
            assert instance.is_running
            value_none = instance.get("[Wheat].Grain.Total.Wt")
            # print("NONE", instance.current_time, value_none)
            # Rewind and run again with full
            instance.rewind()
            instance.rewind(datetime.timedelta(days=20))
            while instance.is_running and not instance.is_complete:
                instance.act("nitrogen", 0.001)
                instance.fast_forward(datetime.timedelta(days=20))
            value_full = instance.get("[Wheat].Grain.Total.Wt")
            # print("FULL", instance.current_time, value_full)
            # Rewind halfway and run without from there
            instance.rewind(time)
            instance.fast_forward()
            instance.fast_forward(datetime.timedelta(days=20))
            value_half = instance.get("[Wheat].Grain.Total.Wt")
            # print("HALF", instance.current_time, value_half)
            # Compare
            assert value_half > value_none
            assert value_full > value_half


def test_env(example_model, apsimx_dir):
    r"""Test of environment creation."""
    logger.setLevel(logging.INFO)
    try:
        env = ApsimXEnv(example_model, apsimx_dir=apsimx_dir)
    finally:
        env.model.stop()
        env.model.model.cleanup()
