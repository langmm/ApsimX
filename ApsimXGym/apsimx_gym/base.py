import os
import copy
import datetime
import contextlib
from collections import defaultdict
from abc import ABC, ABCMeta, abstractmethod
from typing import Optional, Union, Dict, List
from functools import cached_property
import numpy as np
import gymnasium as gym
from . import logger


class RecoverableError(RuntimeError):
    r"""Error that does not stop the engine."""
    pass


class ModelEngineError(RuntimeError):
    r"""Error raised by the model engine."""
    pass


class RecoverableModelEngineError(RecoverableError):
    r"""Error raised by the model engine that does not stop the engine."""
    pass


class InvalidActionError(RecoverableError):
    r"""Error raised when an action is invalid."""
    pass


class BaseModelFile(ABC):
    r"""Base class for managing model input files.

    Args:
        fname: Path to a model file.
        generated: If True, this file was generated.
        contents: Contents to initialize the file with.

    """

    def __init__(self, fname: str, generated: Optional[bool] = False,
                 contents: Optional[dict] = None):
        self.fname = fname
        self.generated = generated
        if contents:
            self.contents = contents

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        r"""Cleanup any generated file."""
        if self.generated and self.exists:
            os.remove(self.fname)
            self.generated = False

    @classmethod
    @abstractmethod
    def _read(cls, fname: str):
        r"""Read a model input file.

        Args:
            fname: Path to file to read.

        Returns:
            object: File contents.

        """
        raise NotImplementedError  # pragma: no cover

    @classmethod
    @abstractmethod
    def _write(cls, fname: str, contents):
        r"""Read a model input file.

        Args:
            fname: Path to file to read.
            contents: File contents to write.

        """
        raise NotImplementedError  # pragma: no cover

    @cached_property
    def contents(self):
        r"""object: File contents."""
        return self._read(self.fname)

    @cached_property
    @abstractmethod
    def is_interactive(self):
        r"""bool: True if the model file is interactive."""
        raise NotImplementedError  # pragma: no cover

    @property
    def exists(self):
        r"""bool: True if the file exists."""
        return os.path.isfile(self.fname)

    def write(self, new_contents: Optional[dict] = None,
              overwrite: Optional[bool] = False):
        r"""Write a new set of contents to the file.

        Args:
            new_contents: New contents to write.
            overwrite: If True, overwrite the existing file.

        """
        if (not overwrite) and os.path.isfile(self.fname):
            raise RuntimeError(f"Model file already exists: "
                               f"\"{self.fname}\"")
        if new_contents is not None:
            self.contents = new_contents
            del self.is_interactive
        self._write(self.fname, self.contents)
        self.generated = True

    def move(self, dst: Optional[str] = None,
             suffix: Optional[str] = None,
             directory: Optional[str] = None) -> str:
        r"""Change the path to the file the contents will be written to
        when write is called.

        Args:
            dst: Path to the new location where the model should be
                saved when write is called.
            suffix: Suffix to add to the current filename if dst is
                not provided.
            directory: Path to the directory that the model should be
                written to when write is called.

        Returns:
            str: The new model file path.

        """
        assert self.contents
        if dst is None and suffix:
            dst = suffix.join(os.path.splitext(self.fname))
        if directory is not None:
            dst = os.path.join(directory, os.path.basename(dst))
        if dst != self.fname:
            self.generated = False
        self.fname = dst
        assert not self.exists
        return self.fname

    def copy(self, **kwargs) -> "BaseModelFile":
        r"""Create a copy of this .apsimx model.

        Args:
            **kwargs: Addiitonal keyword arguments are passed to move.

        Returns:
            ApsimXFile: Copied .apsimx model.

        """
        out = type(self)(self.fname, generated=self.generated)
        out.contents = copy.deepcopy(self.contents)
        if kwargs:
            out.move(**kwargs)
        return out

    def make_interactive(self, **kwargs) -> "BaseModelFile":
        r"""Modify this file to make it interactive.

        Args:
            **kwargs: Keyword arguments are passed to move.

        Returns:
            str: Update model path.

        """
        kwargs.setdefault("suffix", "-Interactive")
        self.move(**kwargs)
        if self.is_interactive:
            logger.warn(f"Source model file \"{self.fname}\" is already "
                        f"interactive")
        else:
            self._make_interactive()
            self.generated = False
        return self.fname

    def set_simulation_times(
            self,
            start_time: Optional[datetime.datetime] = None,
            end_time: Optional[datetime.datetime] = None,
            sow_date: Optional[datetime.datetime] = None,
            harvest_date: Optional[datetime.datetime] = None,
    ):
        r"""Set the simulation start/end time in the file contents.

        Args:
            start_time: Simulation start time.
            end_time: Simulation end time.
            sow_date: Date that the crop should be sown.
            harvest_date: Date that the crop should be harvested.

        """
        if ((start_time is None and end_time is None
             and sow_date is None and harvest_date is None)):
            return
        self._set_simulation_times(start_time, end_time,
                                   sow_date, harvest_date)
        self.generated = False

    @abstractmethod
    def _set_simulation_times(
            self,
            start_time: Optional[datetime.datetime] = None,
            end_time: Optional[datetime.datetime] = None,
            sow_date: Optional[datetime.datetime] = None,
            harvest_date: Optional[datetime.datetime] = None,
    ):
        r"""Set the simulation start/end time in the file contents.

        Args:
            start_time: Simulation start time.
            end_time: Simulation end time.
            sow_date: Date that the crop should be sown.
            harvest_date: Date that the crop should be harvested.

        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def _make_interactive(self):
        r"""Modify this file to make it interactive."""
        raise NotImplementedError  # pragma: no cover


class BaseModelEngine(ABC):
    r"""Base class for exposing a model as an environment engine.

    Args:
        model_file: Path to one or more model input files.
        model_suffix: Additional suffix to add to a copy of the provided
            model file to ensure that it is unique.
        output_dir: Path to the directory where output should be saved.
        start_time: Simulation start time.
        end_time: Simulation end time.
        sow_date: Date that the crop should be sown.
        harvest_date: Date that the crop should be harvested.
        actions: Names of actions to include. Only used if action_map
            not provided.
        action_map: Description of actions available via the act method.

    """

    INPUT_FILE_TYPE = None
    AVAILABLE_ACTION_MAP = {}

    def __init__(
            self,
            model_file: Union[str, List[str]],
            model_suffix: Optional[str] = None,
            output_dir: Optional[str] = None,
            start_time: Optional[datetime.datetime] = None,
            end_time: Optional[datetime.datetime] = None,
            sow_date: Optional[datetime.datetime] = None,
            harvest_date: Optional[datetime.datetime] = None,
            actions: Optional[List[str]] = None,
            action_map: Optional[
                Dict[str, Dict[str, Union[str, float]]]] = None,
    ):
        self.model_file = model_file
        self.model_suffix = model_suffix
        self.output_dir = output_dir
        self.start_time = start_time
        self.end_time = end_time
        self.sow_date = sow_date
        self.harvest_date = harvest_date
        self.action_map = action_map or self.select_actions(actions)
        self.history = defaultdict(lambda: [])
        self.model = self.INPUT_FILE_TYPE(self.model_file)
        self.complete_actions(self.action_map)
        self.update_model_file()

    def update_model_file(self):
        r"""Update the model file to make it interactive and set the
        start/end times."""
        if not self.model.is_interactive:
            self.model.make_interactive()
        if self.output_dir or self.model_suffix:
            self.model.move(directory=self.output_dir,
                            suffix=self.model_suffix)
        if self.start_time or self.end_time:
            if self.model.exists:
                self.model.move(suffix="-Modified")
            self.model.set_simulation_times(
                start_time=self.start_time,
                end_time=self.end_time,
                sow_date=self.sow_date,
                harvest_date=self.harvest_date,
            )
        if not self.model.exists:
            self.model.write()

    @classmethod
    def select_actions(cls, actions: Optional[List[str]] = None,
                       num_levels: Optional[int] = None):
        r"""Select a set of default actions.

        Args:
            actions: Set of actions to select.
            num_levels: Number of levels per action if not specified in
            action_map (0 for continuous).

        Returns:
            dict: Description of selected actions.

        """
        actions = (actions or list(cls.AVAILABLE_ACTION_MAP.keys()))
        return {
            k: cls.AVAILABLE_ACTION_MAP[k].copy() for k in actions
        }

    @classmethod
    def complete_actions(
            cls, action_map: dict,
            num_levels: Optional[int] = 0,
            allow_donothing: bool = True,
            exclusive: bool = True,
    ):
        r"""Complete an action map filling in missing parameters from
        AVAILABLE_ACTION_MAP.

        Args:
            action_map: Descriptions of actions.
            num_levels: Number of levels per action if not specified in
                action_map (0 for continuous).
            allow_donothing: Include non-action as a possible action.
            exclusive: Don't allow more than one action per step.

        """
        for action, desc in action_map.items():
            desc.setdefault("num_levels", num_levels)
            if action in cls.AVAILABLE_ACTION_MAP:
                for k, v in cls.AVAILABLE_ACTION_MAP[action].items():
                    desc.setdefault(k, v)
            if desc["num_levels"] == 1 and "max" not in desc:
                # Boolean
                continue
            if "max" not in desc:
                raise InvalidActionError(
                    f"Error parsing description of action \"{action}\". "
                    f"num_levels = 1 and a \"max\" parameter is not "
                    f"provided, indicating a non-boolean action. "
                    f"Non-boolean actions must have a maximum value."
                )
            if desc["num_levels"] == 0:
                desc.setdefault("min", 0.0)
            elif "levels" not in desc:
                if allow_donothing and not exclusive:
                    desc.setdefault("min", 0.0)
                else:
                    desc.setdefault(
                        "min", desc["max"] / desc["num_levels"])
                desc["levels"] = np.linspace(
                    desc["min"], desc["max"], desc["num_levels"]
                ).tolist()

    @property
    def is_complete(self):
        r"""bool: True if the simulation is complete."""
        return self.current_time >= self.end_time

    def __del__(self):
        self.stop()

    @property
    @abstractmethod
    def is_running(self):
        r"""bool: True if the model engine is still running."""
        raise NotImplementedError  # pragma: no cover

    @property
    def is_operable(self):
        r"""bool: True if the model engine is running and functioning."""
        return self.is_running

    @property
    @abstractmethod
    def current_time(self):
        r"""datetime.datetime: Current simulation time."""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def start(self):
        r"""Start the model engine."""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def stop(self):
        r"""Stop the model engine."""
        raise NotImplementedError  # pragma: no cover

    def reset(self):
        r"""Re-start the model."""
        self.stop()
        self.start()
        self.history = defaultdict(lambda: [])

    @abstractmethod
    def _get(self, name: str):
        r"""Send a request to get the current value of a simulation state
        variable.

        Args:
            name: Name of variable to get the value of.

        Returns:
            object: Current variable value.

        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def _set(self, name: str, value):
        r"""Send a request to set a simulation state variable.

        Args:
            name: Name of the variable to update.
            value: New value for the named variable.

        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def _act(self, action: str, param: dict):
        r"""Perform an action.

        Args:
            name: Name of the action to perform.
            param: Action parameters.

        """
        raise NotImplementedError  # pragma: no cover

    @contextlib.contextmanager
    def stop_on_error(self, record: Optional[tuple] = None,
                      allow_error: Optional[bool] = False):
        r"""Context manager that stops the simulation on an error.

        Args:
            record: Action to log when successful.
            allow_error: If True, a RecoverableError error will not
                result in the simulation being stopped.

        """
        try:
            yield
            if record:
                self.record(*record)
        except RecoverableError as e:
            if allow_error:
                logger.warning(e)
            else:
                raise
        except BaseException:
            self.stop()
            raise

    def get(self, name: str, allow_error: Optional[bool] = False):
        r"""Send a request to get the current value of a simulation state
        variable.

        Args:
            name: Name of variable to get the value of.
            allow_error: If True, a RecoverableError error will not
                result in the simulation being stopped.

        Returns:
            object: Current variable value.

        """
        out = None
        with self.stop_on_error(allow_error=allow_error):
            out = self._get(name)
        logger.debug(f"get: {name} -> {out}")
        return out

    def set(self, name: str, value, allow_error: Optional[bool] = False):
        r"""Send a request to set a simulation state variable.

        Args:
            name: Name of the variable to update.
            value: New value for the named variable.
            allow_error: If True, a RecoverableError error will not
                result in the simulation being stopped.

        """
        out = None
        with self.stop_on_error(("set", name, (value, ), {}),
                                allow_error=allow_error):
            out = self._set(name, value)
        logger.debug(f"set: {name} -> {value}")
        return out

    def act(self, action: str, *args,
            allow_error: Optional[bool] = False,
            **kwargs):
        r"""Perform an action.

        Args:
            name: Name of the action to perform.
            *args: Additional positional arguments provide action
                 parameters in the order specified by \"param\" in the
                 action_map.
            allow_error: If True, a RecoverableError error will not
                result in the simulation being stopped.
            **kwargs: Additional keyword arguments provide action
                 parameters by name.

        """
        out = None
        with self.stop_on_error(("act", action, args, kwargs),
                                allow_error=allow_error):
            if action not in self.action_map and action != "terminate":
                raise InvalidActionError(
                    f"Unsupported action \"{action}\". Supported "
                    f"actions include: {list(self.action_map.keys())}")
                # return self.set(action, *args)
            param = self.action_map.get(action, {}).get("param", [])
            if len(args) > len(param):
                raise InvalidActionError(
                    f"Tool many ({len(args)}) parameters provided for "
                    f"action ({action}). Valid parameters: {param}")
            kws = {}
            for k, v in kwargs.items():
                if k not in param:
                    raise InvalidActionError(
                        f"Invalid parameter \"{k}\" provided for action "
                        f"\"{action}\". Valid parameters: {param}")
                kws[k] = v
            for k, v in zip(param, args):
                if k in kws:
                    raise InvalidActionError(
                        f"\"{k}\" parameter for action \"{action}\" "
                        f"provided as both a positional and keyword "
                        f"argument")
                kws[k] = v
            out = self._act(action, kws)
        logger.debug(f"act: {action}[{args}, {kwargs}]")
        if action == "terminate":
            self.resume(wait=True)
        return out

    def getvars(self, names: list, allow_error: Optional[bool] = False):
        r"""Send a request to get the current value of a set of
        simulation state variables.

        Args:
            names: Names of variables to get values for.
            allow_error: If True, a RecoverableError error will not
                result in the simulation being stopped.

        Returns:
            dict: Mapping between state variable names and retrieved
                values.

        """
        out = {}
        for name in names:
            out[name] = self.get(name, allow_error=allow_error)
        return out

    def setvars(self, values: dict, allow_error: Optional[bool] = False):
        r"""Send a request to set simulation state variables.

        Args:
            values: Mapping between state variable names and the
                values they should be set to.
            allow_error: If True, a RecoverableError error will not
                result in the simulation being stopped.

        """
        for k, v in values.items():
            self.set(k, v, allow_error=allow_error)

    def actvars(self, values: dict, allow_error: Optional[bool] = False):
        r"""Perform multiple actions.

        Args:
            values: Mapping between action names and tuples of
                action parameters.
            allow_error: If True, a RecoverableError error will not
                result in the simulation being stopped.

        """
        for k, v in values.items():
            self.act(k, *v, allow_error=allow_error)

    def record(self, *args):
        r"""Record an action.

        Args:
            *args: Positional arguments are stored in the history.

        """
        self.history[self.current_time].append(args)

    def fast_forward(
            self, time: Optional[Union[datetime.datetime,
                                       datetime.timedelta]] = None
    ):
        r"""Fast forward the simulation to the desired time.

        Args:
            time: Time that simulation should be run to or the the
                time that the simulation should be run for (timedelta).

        """
        if time is None:
            time = self.end_time
        elif isinstance(time, datetime.timedelta):
            time = min(self.current_time + time, self.end_time)
        if time <= self.current_time:
            # if self.current_time == self.end_time:
            #     self.resume()  # Ensure that the simulation finishes
            return
        if time > self.end_time:
            logger.warning(f"Cannot fast-forward to {time} from "
                           f"{self.current_time}. It exceeds "
                           f"the simulation end time {self.end_time}")
            time = self.end_time
        if time <= self.current_time:
            # if self.current_time == self.end_time:
            #     self.resume()
            return
        logger.info(f"Fast-forward to {time} from {self.current_time}")
        while ((self.is_running and self.current_time <= time
                and not self.is_complete)):
            self.resume(wait=True)

    def rewind(self, time: Optional[Union[datetime.datetime,
                                          datetime.timedelta]] = None):
        r"""Rewind the simulation to a previous time.

        Args:
            time: Time to rewind to or time to rewind by (timedelta).

        """
        if time is None:
            time = self.start_time
        elif isinstance(time, datetime.timedelta):
            time = max(self.current_time - time, self.start_time)
        if time < self.start_time:
            logger.warning(f"Cannot rewind to {time} from "
                           f"{self.current_time}. It preceeds "
                           f"the simulation start time {self.start_time}")
            time = self.start_time
        if time >= self.current_time:
            return
        logger.info(f"Rewinding to {time} from {self.current_time}")
        history = self.history
        self.reset()
        for t, actions in history.items():
            if t > time:
                break
            self.fast_forward(t)
            for action in actions:
                logger.info(f"Replaying t={t}: {action}")
                getattr(self, action[0])(
                    action[1], *action[2], **action[3])
        if time > self.current_time:
            self.fast_forward(time)

    @abstractmethod
    def resume(self, wait: Optional[bool] = False):
        r"""Resume the simulation.

        Args:
            wait: If True, wait for the simulation to pause.

        """
        raise NotImplementedError  # pragma: no cover


class BaseModelEnv(gym.Env, metaclass=ABCMeta):
    r"""ApsimX environment.

    Args:
        model_file: Path to one or more model input files.
        start_time: Simulation start time.
        end_time: Simulation end time.
        intervention_interval: Time between decisions. If an integer is
            provided, the units will be assumed to be days.
        output_vars: List of observation variable names.
        num_levels: Number of levels per action if not specified in
            action_map (0 for continuous).
        actions: Names of actions to include. Only used if action_map
            not provided.
        action_map: Custom description mapping for actions.
        revenue_var: Description of how profit should be calculated from
            an output variable.
        param: Additional parameters to set when the simulation begins.
        allow_donothing: Include non-action as a possible action.
        exclusive: Don't allow more than one action per step.
        **kwargs: Additional keyword arguments are passed to the model
            engine constructor.

    """

    MODEL_ENGINE_CLASS = None
    DEFAULT_ACTIONS = []
    DEFAULT_REVENUE_VAR = {}

    def __init__(
            self,
            model_file: str,
            start_time: datetime.datetime = None,
            end_time: datetime.datetime = None,
            intervention_interval: Union[int, datetime.timedelta] = 7,
            output_vars: Optional[List[str]] = None,
            num_levels: int = 4,
            actions: Optional[List[str]] = None,
            action_map: Optional[
                Dict[str, Dict[str, Union[str, float]]]] = None,
            revenue_var: Optional[Dict[str, Union[str, float]]] = None,
            param: Optional[dict] = None,
            allow_donothing: bool = True,
            exclusive: bool = True,
            **kwargs
    ):
        self.model_file = model_file
        self.start_time = start_time
        self.end_time = end_time
        self.intervention_interval = (
            datetime.timedelta(intervention_interval)
            if isinstance(intervention_interval, int)
            else intervention_interval
        )
        self.output_vars = output_vars or []
        self.num_levels = num_levels
        self.action_map = (
            action_map or self.MODEL_ENGINE_CLASS.select_actions(
                actions or self.DEFAULT_ACTIONS)
        )
        self.revenue_var = revenue_var or copy.deepcopy(
            self.DEFAULT_REVENUE_VAR)
        if self.revenue_var:
            self.output_vars = [self.revenue_var["name"]] + [
                k for k in self.output_vars
                if k != self.revenue_var["name"]
            ]
        self.param = param
        self.allow_donothing = allow_donothing
        self.exclusive = exclusive
        self.model_kwargs = kwargs
        self.MODEL_ENGINE_CLASS.complete_actions(
            self.action_map, num_levels=self.num_levels,
            allow_donothing=self.allow_donothing,
            exclusive=self.exclusive,
        )

        # Define what actions are available (bounds for each parameter)
        self.action_order = []
        if self.exclusive:
            if all("levels" in desc for desc in
                   self.action_map.values()):
                if self.allow_donothing:
                    self.action_order.append({})
                for action, desc in self.action_map.items():
                    self.action_order += [
                        {action: value} for value in desc["levels"]
                    ]
                self.action_space = gym.spaces.Discrete(
                    int(self.allow_donothing)
                    + sum(len(desc["levels"]) for desc in
                          self.action_map.values())
                )
            else:
                if self.allow_donothing:
                    self.action_order.append(None)
                self.action_order += list(self.action_map.keys())
                self.action_space = gym.spaces.OneOf([
                    self._create_action_space(
                        self.action_map.get(action, {}))
                    for action in self.action_order
                ])
        else:
            self.action_space = gym.spaces.Dict(
                {
                    action: self._create_action_space(desc)
                    for action, desc in self.action_map.items()
                }
            )
            self.action_order = list(self.action_map.keys())

        # Define what the agent can observe (bounds for each output)
        self.observation_space = gym.spaces.Box(
            shape=(1 + len(self.output_vars), ),
            low=-np.inf, high=np.inf,
            dtype=np.float64,
        )

        self.model = self.create_model(**self.model_kwargs)
        self.model.start()
        if self.param:
            self.model.setvars(self.param)
        if self.start_time is None:
            self.start_time = self.model.start_time
        if self.end_time is None:
            self.end_time = self.model.end_time
        self.log = self._init_log()

    @property
    def current_time(self):
        r"""datetime.dateime: Current time."""
        return self.model.current_time

    @property
    def field_area(self):
        r"""float: Field area."""
        return 1.0  # Report in terms of $/ha

    @property
    def intervention_timedelta(self):
        r"""datetime.timedelta: Intervention interval as delta."""
        if isinstance(self.intervention_interval, datetime.timedelta):
            return self.intervention_interval
        return datetime.timedelta(self.intervention_interval)

    @classmethod
    def _create_action_space(cls, desc):
        if not desc:
            return gym.spaces.Discrete(1)
        if "levels" in desc:
            return gym.spaces.Discrete(len(desc["levels"]))
        return gym.spaces.Box(shape=(1,), dtype=np.float64,
                              low=desc["min"], high=desc["max"])

    def _wrap_action(self, action):
        if isinstance(self.action_space, gym.spaces.Discrete):
            return self.action_order[action]
        elif isinstance(self.action_space, gym.spaces.OneOf):
            space = self.action_order[action[0]]
            if space is None:
                return {}
            if "levels" in self.action_map[space]:
                value = self.action_map[space][action[1]]
            else:
                value = action[1]
            return {space: value}
        out = {}
        if not isinstance(action, dict):
            action = {k: v for k, v in zip(self.action_order, action)}
        for k, v in action.items():
            if "levels" in self.action_map[k]:
                out[k] = self.action_map[k]["levels"][v]
            else:
                out[k] = v
        return out

    def _init_log(self) -> dict:
        r"""Initialize the log."""
        return {
            "action": dict(),
            "obs": dict(),
            "cost": dict(),
            "reward": dict(),
            "day": dict(),
        }

    def _log(self, action: dict, obs: dict, reward: float) -> None:
        """Log the outputs into the log dictionary

        Args:
            obs: The observation
            action: The action taken by the agent
            reward: The reward

        """
        self.log["action"][self.current_time] = action
        self.log["obs"][self.current_time] = obs
        self.log["cost"][self.current_time] = self._get_cost(action)
        self.log["reward"][self.current_time] = reward
        self.log["day"][self.current_time] = self.current_time

    def create_model(self, **kwargs) -> BaseModelEngine:
        r"""Create a new model engine."""
        return self.MODEL_ENGINE_CLASS(
            self.model_file,
            start_time=self.start_time,
            end_time=self.end_time,
            action_map=self.action_map,
            **kwargs
        )

    def close(self):
        r"""Close the environment."""
        self.model.stop()
        super().close()

    def _get_cost(self, action: dict) -> float:
        r"""Calculate the cost of the current action.

        Args:
            action: Current action.

        Returns:
            float: Action cost.

        """
        out = 0.0
        for k, v in action.items():
            if self.action_map[k].get("cost", 0) > 0:
                out += (
                    self.action_map["cost"] * v
                    * self.model.field_area
                )
        return out

    def _get_revenue(self, obs: dict) -> float:
        r"""Calculate the revenue based on the current observation.

        Args:
            obs: Current observation.

        Returns:
            float: Revenue.

        """
        if not self.revenue_var:
            raise NotImplementedError(
                "Provide revenue_var or override the _get_reward method")
        return (
            obs[self.reveue_var["name"]] * self.reveue_var.get("cost", 1)
            * self.model.field_area
        )

    def _get_reward(self, action: dict, observation) -> float:
        r"""Calculate the reward from the current observation.

        Args:
            observation: Observation to calculate the reward for.

        Returns:
            float: Reward value.

        """
        if not self.revenue_var:
            raise NotImplementedError(
                "Provide revenue_var or override the _get_reward method")
        revenue = self._get_revenue(observation)
        cost = sum(v for v in self.log["cost"]) + self._get_cost(action)
        return revenue - cost

    def _get_obs(self) -> np.ndarray:
        r"""Observe the current state of the model.

        Returns:
            np.ndarray: Array of state parameters.

        """
        return self.model.getvars(self.output_vars)

    def _process_observation(self, observation: dict):
        r"""Force the observations into the expected format.

        Args:
            observation: Raw observations.

        Returns:
            np.ndarray: Array of observations parameters.

        """
        days_elapsed = self.model.current_time - self.model.start_time
        observation = np.concatenate([
            np.array(list(observation.values())),
            [days_elapsed.days],
        ])
        for i in range(len(observation)):
            if isinstance(observation[i], datetime.date):
                observation[i] = int(observation[i].strftime("%Y%m%d"))
            if isinstance(observation[i], str):
                observation[i] = 0
        return observation.astype("float64")

    def reset(self, seed: Optional[int] = None,
              options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        r"""Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration

        Returns:
            tuple: (observation, info) for the initial state

        """
        # IMPORTANT: Must call this first to seed the random number generator
        # TODO: Process options
        super().reset(seed=seed)
        self.log = self._init_log()
        self.model.reset()
        observation = self._get_obs()
        return self._process_observation(observation), self.log

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one timestep within the environment.

        Args:
            action: The action to take (modification of state variable).

        Returns:
            tuple: (observation, reward, terminated, truncated, info)

        """
        action_dict = self._wrap_action(action)
        self.model.actvars(action_dict)
        self.model.fast_forward(self.intervention_timedelta)
        observation = self._get_obs()
        reward = self._get_reward(action_dict, observation)
        terminated = (not self.model.is_running)
        truncated = self.model.is_complete
        self._log(action, observation, reward)
        return (self._process_observation(observation), reward,
                terminated, truncated, self.log)
