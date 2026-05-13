import os
import re
import copy
import uuid
import pprint
import datetime
import contextlib
from collections import defaultdict
from abc import ABC, ABCMeta, abstractmethod
from typing import Optional, Union, Dict, List, Tuple, Any, Callable
from functools import cached_property
import numpy as np
import gymnasium as gym
from . import logger


class NoDefault:
    r"""Dummy class for defaults."""
    pass


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


def readonly_cached_property(method: Callable):
    r"""Decorator for a read-only cached property.

    Args:
        method: Method to wrap.

    """

    name = method.__qualname__.rsplit('.', 1)[-1]

    @property
    def _readonly_cached_property(self):
        if name not in self._cached_properties:
            self._cached_properties[name] = method(self)
        return self._cached_properties[name]

    return _readonly_cached_property


class CachedPropertyMixin:
    r"""Mixin class for enabling read-only cached properties."""

    def __init__(self, *args, **kwargs):
        self._cached_properties = {}
        super().__init__(*args, **kwargs)

    def _clear_cached_property(self, name: str):
        self._cached_properties.pop(name, None)

    def _clear_cached_properties(self):
        self._cached_properties.clear()


class ModelAction(CachedPropertyMixin):
    r"""Wrapper for a model action.

    Args:
        name: Action name.
        alias: Action alias.
        keywords: Key words or phrases identifying this action.
        cost: Action cost. If the action produces a float, this should be
            the cost per action unit.
        action_param: Parameter that action will set.
        num_levels: Number of levels that the action supports for the
            action parameter. 0 indicates a continuous action, -1
            indicates a boolean action.
        level: Explicit levels for the action parameter.
        bounds: Explicit bounds for the action parameter (numbers only).
        param_desc: Descriptions of parameters supported by the action.
        param: Values for additional parameters that should be used.
        allow_donothing: If True, the action should allow for a choice to
            do nothing.
        offset: Action offset when part of a discrete set.

    """
    ACTION_SCHEMA = {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "alias": {"type": "string"},
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
            },
            "cost": {"type", "number"},
            "action_param": {
                "type": ["string", "null"],
            },
            "num_levels": {"type": "integer"},
            "levels": {
                "type": "array",
                "items": {"type": ["number", "string"]},
            },
            "bounds": {
                "type": "array",
                "items": [{"type": "number"}, {"type": "number"}],
            },
            "param_desc": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",  # "schema" for yggdrasil_rapidjson
                    "properties": {
                        "type": {"type": "string"},
                        "enum": {"type": "array"},
                        "min": {"type": "number"},
                        "max": {"type": "number"},
                        "units": {"type": "string"},
                        "ndim": {"type": "integer"},
                        "dtype": {"type": "string"},
                    },
                },
            },
            "param": {"type": "object"},
        },
        "additionalProperties": False,
        "required": ["description", "action_param", "num_levels"],
    }

    def __init__(
            self,
            name: str, description: str,
            alias: Optional[str] = None,
            keywords: Optional[List[str]] = None,
            cost: Optional[float] = None,
            action_param: Optional[str] = None,
            num_levels: Optional[int] = 0,
            levels: Optional[List[Union[str, float, np.ndarray]]] = None,
            bounds: Optional[Union[Tuple[float, float],
                                   Tuple[np.ndarray, np.ndarray]]] = None,
            param_desc: Optional[dict] = None,
            param: Optional[dict] = None,
            allow_donothing: Optional[bool] = True,
            offset: Optional[int] = 0,
    ):
        self.name = name
        self.description_fstring = description
        self.alias = alias
        self.keywords = keywords.copy() if keywords else []
        self.cost = cost
        self.action_param = action_param
        self.action_param_desc = None
        self.param_desc = param_desc or {}
        self.param = {}
        self.allow_donothing = allow_donothing
        self.offset = offset
        self.num_levels = len(levels) if levels else num_levels
        if self.num_levels != -1:
            assert action_param is not None
            self.action_param_desc = self.param_desc[action_param]
            if "enum" in self.action_param_desc and levels is None:
                levels = self.action_param_desc["enum"]
                self.num_levels = len(levels)
        self._levels = levels
        self._bounds = bounds
        if param:
            self.set_param(param)
        super().__init__()

    @readonly_cached_property
    def additional_param(self):
        r"""list: Set of additional parameters."""
        return [k for k in self.param_desc.keys()
                if k != self.action_param]

    @readonly_cached_property
    def additional_param_args(self) -> list:
        r"""Set of additional parameter arguments."""
        if not self.param:
            return []
        out = [self.param.get(k, self.param_desc[k].get("default", None))
               for k in self.additional_param]
        while out[-1] is None:
            out = out[:-1]
        if None in out:
            idx_first = out.index(None)
            missing = [
                self.additional_param[i - 1]
                for i, x in enumerate(out) if x is None
            ]
            later = [
                self.additional_param[i - 1]
                for i, x in enumerate(out)
                if x is not None and i > idx_first
            ]
            raise InvalidActionError(
                f"Missing optional parameters ({missing}) "
                f"required to be able to provide those "
                f"that occur later in the order ({later})"
            )
        return out

    @readonly_cached_property
    def bounds(self):
        r"""tuple: Minimum and maximum bounds for action value."""
        if self._bounds:
            return self._bounds
        if self.num_levels == -1:
            return None
        if not self.numeric:
            return None
        if self._bounds:
            xmin, xmax = self._bounds[:]
        else:
            xmin = self.action_param_desc.get("min", -np.inf)
            xmax = self.action_param_desc.get("max", np.inf)
        le = xmin
        re = xmax
        if not isinstance(le, np.ndarray):
            le = np.empty(self.shape, dtype=self.dtype)
            le.fill(xmin)
        if not isinstance(re, np.ndarray):
            re = np.empty(self.shape, dtype=self.dtype)
            re.fill(xmax)
        return (le, re)

    @readonly_cached_property
    def levels(self):
        r"""list: Set of discrete levels for the action."""
        if self._levels:
            return self._levels
        if self.num_levels == 0 or self.num_levels == -1:
            return None
        if any(any(x == -np.inf) or any(x == np.inf)
               for x in self.bounds):
            raise InvalidActionError(
                f"Error parsing description of action \"{self.name}\". "
                f"Cannot create discrete levels for an infinite "
                f"action space. The bounds for action space parameter "
                f"\"{self.action_param}\" are {self.bounds}."
            )
        if (not self.allow_donothing) and all(self.bounds[0] == 0):
            out = np.linspace(
                self.bounds[1] / self.num_levels,
                self.bounds[1], self.num_levels
            ).tolist()
        else:
            out = np.linspace(
                self.bounds[0], self.bounds[1], self.num_levels
            ).tolist()
        return [
            np.array(x, dtype=self.dtype)
            for x in out
        ]

    @readonly_cached_property
    def numeric(self) -> bool:
        r"""bool: True if the action is numeric."""
        return (self.action_param_desc
                and self.action_param_desc["type"] == "number")

    @readonly_cached_property
    def ndim(self) -> int:
        r"""int: Number of dimensions in the action parameter."""
        if not self.numeric:
            return None
        return self.action_param_desc.get("ndim", 1)

    @property
    def shape(self) -> tuple:
        r"""tuple: Shape of action parameter."""
        return (self.ndim, )

    @readonly_cached_property
    def dtype(self) -> type:
        r"""dtype: Data type of the action parameter."""
        if not self.numeric:
            return None
        return self.action_param_desc.get("dtype", np.float64)

    @readonly_cached_property
    def choices(self) -> list:
        r"""list: Set of choices."""
        if self.num_levels == 0:
            return []
        elif self.num_levels == -1:
            return [False, True] if self.allow_donothing else [True]
        # TODO: Add do nothing case for strings?
        return self.levels

    def set_param(self, param: dict,
                  src: Optional[str] = "set_param"):
        r"""Update the action parameters.

        Args:
            param: Action parameters.
            src: Description of how the parameter is being updated for
                logging parameter conflicts.

        """
        invalid = []
        for k, v in param.items():
            if k not in self.param_desc or k == self.action_param:
                invalid.append(k)
                continue
            if k in self.param and self.param[k] != v:
                logger.warning(
                    f"Parameter \"{k}\" specified via {src} "
                    f"conflicts with the action parameter for the "
                    f"\"{self.name}\" action. The {src} value {v} "
                    f"will be used and the action parameter value "
                    f"{self.param[k]} will be discarded"
                )
            self.param[k] = v
        if invalid:
            raise KeyError(f"Invalid parameters provided for action "
                           f"\"{self.name}\": {invalid}")
        assert self.action_param not in self.param
        self._clear_cached_properties()

    def scale_action_amounts(self, scale: Union[int, float]):
        r"""Scale action limits/levels.

        Args:
            scale: Amount to scale values by.

        """
        assert self.numeric
        if self._levels:
            self._levels = [scale * x for x in self._levels]
        else:
            self._bounds = (self.bounds[0], scale * self.bounds[1])
        self._clear_cached_properties()

    @property
    def num_choices(self) -> int:
        r"""int: Number of discrete choices allowed for this action."""
        return len(self.choices)

    @property
    def discrete(self) -> bool:
        r"""bool: True if the action is discrete."""
        return (self.num_levels != 0)

    @property
    def boolean(self) -> bool:
        r"""bool: True if the action is boolean."""
        return (self.num_levels == -1)

    @readonly_cached_property
    def example_value(self):
        r"""object: Example action value."""
        if self.num_levels == 0:
            for x in self.bounds[::-1]:
                if not (any(x == -np.inf) or any(x == np.inf)):
                    return x
            return 0.0
        elif self.num_levels == -1:
            return True
        return self.levels[-1]

    @readonly_cached_property
    def example_args(self) -> tuple:
        r"""tuple: Example action args."""
        return self.value2args(self.example_value)

    @readonly_cached_property
    def example_description(self) -> str:
        r"""str: Example action description."""
        return self.format_description(value=self.example_value)

    @readonly_cached_property
    def description(self) -> str:
        r"""str: Description of the action set."""
        return self.format_description()

    @readonly_cached_property
    def description_regex(self) -> str:
        r"""str: Regex string for parsing descriptions."""
        return self.format_description_regex()

    @readonly_cached_property
    def space(self) -> gym.spaces.space.Space:
        r"""gym.spaces.space.Space: Action space."""
        if self.num_levels == 0:
            return gym.spaces.Box(
                shape=self.shape,
                dtype=self.dtype,
                low=self.bounds[0],
                high=self.bounds[1],
            )
        return gym.spaces.Discrete(self.num_choices)

    def combine_args_and_kwargs(self, args: tuple, kwargs: dict) -> dict:
        r"""Combine positional and keyword arguments into a single dict
        for the action based on the available action parameters.

        Args:
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            dict: Combined keyword arguments.

        """
        if len(args) > len(self.param_desc):
            raise InvalidActionError(
                f"Tool many ({len(args)}) parameters provided for "
                f"action ({self.name}). Valid parameters: "
                f"{list(self.param_desc.keys())}")
        kws = {}
        for k, v in kwargs.items():
            if k not in self.param_desc:
                raise InvalidActionError(
                    f"Invalid parameter \"{k}\" provided for action "
                    f"\"{self.name}\". Valid parameters: "
                    f"{list(self.param_desc.keys())}")
            kws[k] = v
        for k, v in zip(self.param_desc.keys(), args):
            if k in kws:
                raise InvalidActionError(
                    f"\"{k}\" parameter for action \"{self.name}\" "
                    f"provided as both a positional and keyword "
                    f"argument")
            kws[k] = v
        for k, v in self.param.items():
            kws.setdefault(k, v)
        # TODO: Validate param against schema?
        # for k, v in kws.items():
        #     rj.validate(self.param_desc[k], v)
        return kws

    def args2cost(self, args: tuple):
        r"""Convert a set of action arguments to the action cost.

        Args:
            args: Action arguments.

        Returns:
            float: Cost of the action.

        """
        # TODO: Allow different costs for different string choices or
        #   optional param?
        if self.cost is None:
            return 0.0
        if not (self.action_param_desc
                and self.action_param_desc["type"] == "number"):
            return self.cost
        return self.cost * args[0]

    def description2action(self, description: str):
        r"""Parse a description to get an action ID.

        Args:
            description: Action description.

        Returns:
            object: Action ID.

        """
        return self.value2action(self.description2value(description))

    def search_description(self, description: str) -> re.Match:
        r"""Search a description for a match to this action using regex.

        Args:
            description: Action description.

        Returns:
            re.Match: Search result.

        """
        return re.search(self.description_regex, description)

    def fuzzy_search_description(self, description: str):
        r"""Search a description for a match to this action by looking
        for keywords.

        Args:
            description: Action description.

        Returns:
            object: Value from fuzzy search.

        """
        desc = description.lower()
        if (((self.numeric or self.boolean)
             and (self.name.lower() in desc
                  or any(k.lower() in desc for k in self.keywords)))):
            if self.boolean:
                return True
            amount_match = re.search(r"(\d+\.?\d*)", description)
            if amount_match:
                return self.dtype(amount_match.group(1))
        raise InvalidActionError(f"Failed to parse description "
                                 f"via fuzzy search: "
                                 f"\"{description}\"")

    def match2value(self, match: re.Match):
        r"""Convert a regex search result into an action value.

        Args:
            match: Regex search result.

        Returns:
            object: Action value.

        """
        if match:
            if self.boolean:
                return True
            value = match.groupdict()[self.action_param]
            if self.numeric:
                if self.ndim == 1:
                    value = self.dtype(value)
                else:
                    value = np.array([
                        self.dtype(x) for x in
                        value.strip("[").strip("]").split(",")
                    ])
            return value
        if self.boolean and False in self.choices:
            return False
        raise InvalidActionError("No match")

    def description2value(self, description: str):
        r"""Parse a description for a action value.

        Args:
            description: Action description.

        Returns:
            object: Action value.

        """
        match = self.search_description(description)
        if match:
            return self.match2value(match)
        raise InvalidActionError(f"Failed to parse description "
                                 f"via regex: \"{description}\"")

    def value2action(self, value):
        r"""Convert an action value into an action ID.

        Args:
            value: Action value.

        Returns:
            int, np.ndarray: Action ID.

        """
        if self.num_levels == 0:
            if not isinstance(value, np.ndarray):
                value = np.array([value], dtype=self.dtype)
            assert value.shape == self.shape
            assert value.dtype == self.dtype
            return value
        if value in self.choices:
            return self.choices.index(value) + self.offset
        if isinstance(value, (float, int, np.ndarray)) and self.numeric:
            diff = np.array(self.choices) - value
            diff = (
                np.abs(diff) if self.ndim == 1
                else np.linalg.norm(diff, axis=1)
            )
            assert len(diff) == self.ndim
            return self.offset + np.argmin(diff)
        raise InvalidActionError(f"{value} is not a valid choice")

    def action2value(self, action: Union[int, np.ndarray]):
        r"""Convert an action ID into a parameter value.

        Args:
            action: Action ID.

        Returns:
            object: Parameter value.

        """
        if self.num_levels == 0:
            if not isinstance(action, np.ndarray):
                raise InvalidActionError(
                    f"Continuous action {self.name} requires a "
                    f"np.ndarray action ID, not {type(action)}"
                 )
            if action.dtype != self.dtype:
                raise InvalidActionError(
                    f"Continuous action {self.name} requires a "
                    f"np.ndarray action ID with dtype {self.dtype}, "
                    f"not {action.dtype}"
                )
            if action.shape != self.shape:
                raise InvalidActionError(
                    f"Continuous action {self.name} requires a "
                    f"np.ndarray action ID "
                    f"with shape {self.shape}, not {action.shape}"
                )
            return action
        if not isinstance(action, (int, np.integer)):
            raise InvalidActionError(
                f"Discrete action {self.name} requires an integer "
                f"action ID, not {type(action)}"
            )
        action_rel = action - self.offset
        if action_rel < 0 or action_rel >= self.num_choices:
            raise InvalidActionError(
                f"Discrete action {self.name} requires an integer "
                f"with the range [{self.offset}, "
                f"{self.offset + self.num_choices}), "
                f"not {action}"
            )
        return self.choices[action_rel]

    def value2args(self, value) -> tuple:
        r"""Convert an action value to arguments.

        Args:
            value: Action value.

        Returns:
            tuple: Action arguments.

        """
        if self.boolean:
            assert value is True
            return tuple([] + self.additional_param_args)
        if self.numeric and isinstance(value, np.ndarray):
            return tuple(value.tolist() + self.additional_param_args)
        return tuple([value] + self.additional_param_args)

    def action2description(self, action) -> str:
        r"""Convert an action ID into a natural language description.

        Args:
            action: Action ID.

        Returns:
            str: Action description.

        """
        return self.format_description(value=self.action2value(action))

    def action2args(self, action) -> tuple:
        r"""Convert an action ID into arguments that can be passed to
        BaseModelEngine.act.

        Args:
            action: Action ID.

        Returns:
            tuple: Parameter act arguments.

        """
        return self.value2args(self.action2value(action))

    @classmethod
    def _format_value(cls, value) -> str:
        if isinstance(value, str):
            return value
        elif isinstance(value, float):
            return f"{value:.1f}"
        elif isinstance(value, list):
            return cls._format_choice_list(value)
        elif isinstance(value, np.ndarray):
            if len(value) == 1:
                return cls._format_value(value[0])
            return (
                "[" + ", ".join(cls._format_value(x) for x in value)
                + "]"
            )
        raise NotImplementedError(type(value))

    @classmethod
    def _format_choice_list(cls, values: list) -> str:
        if not values:
            return ""
        values = [cls._format_value(v) for v in values]
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return f"{values[0]} or {values[1]}"
        return f"{', '.join(values[:-1])}, or {values[-1]}"

    def format_description(
            self, value=None,
            param: Optional[dict] = None,
    ) -> str:
        r"""Format a description of the action.

        Args:
            value: Action value to include in the description.
            param: Alternate action parameter values to include in the
                description.

        Returns:
            str: Formatted action description.

        """
        if value is None:
            if self.num_levels == 0:
                if self.ndim == 1:
                    value = f"{self.bounds[0][0]} to {self.bounds[1][0]}"
                else:
                    value = f"{self.bounds[0]} to {self.bounds[1]}"
            elif self.num_levels == -1:
                value = True
            else:
                value = self.levels
        kws = dict(self.param, **(param or {}))
        if self.num_levels == -1:
            if value is False:
                return ""
            assert value in [None, True]
        else:
            kws.setdefault(self.action_param, value)
        for k, v in self.param_desc.items():
            if "units" in v:
                kws.setdefault(f"{k}_units", v["units"])
            kws.setdefault(k, v.get("default", ""))
        out = self.description_fstring.format(**{
            k: self._format_value(v)
            for k, v in kws.items()
        })
        while "  " in out:
            out = out.replace("  ", " ")  # Collapse extra whitespace
        if not out.endswith("."):
            out += "."
        return out

    @classmethod
    def _param2regex(cls, desc: dict) -> str:
        if desc["type"] == "string":
            if "enum" in desc:
                return "|".join(f"(?:{x})" for x in desc["enum"])
            return ".+?"
        elif desc["type"] == "number":
            out = r"\d+(?:\.\d+)?"
            ndim = desc.get("ndim", 1)
            if ndim > 1:
                out = r"\[\s*" + r"\s*\,\s*".join(ndim * [out]) + r"\s*\]"
            return out
        raise NotImplementedError(
            f"Regex for parameter type {desc['type']}")

    def format_description_regex(
            self,
            param: Optional[dict] = None,
            param_regex: Optional[dict] = None,
    ) -> str:
        r"""Create a regex string for extracting parameters from an
        action description.

        Args:
            param: Parameter values that should be included in the
                description regex as constants.
            param_regex: Regex strings for parameters that should be
                matched in the description regex.

        Returns:
            str: Description regex.

        """
        param = dict(
            self.param, **(param.copy() if param is not None else {})
        )
        param_regex = param_regex.copy() if param_regex else {}
        placeholders = {}
        for k, v in self.param_desc.items():
            if k in param or k in param_regex:
                continue
            param_regex[k] = self._param2regex(v)
        for k, v in param_regex.items():
            kph = k.upper()
            assert kph not in self.description_fstring
            placeholders[kph] = '(?P<' + k + '>' + v + ')'
            param[k] = kph
        out = re.escape(self.format_description(param=param))
        for k, v in placeholders.items():
            out = out.replace(k, v)
        return out

    @readonly_cached_property
    def constraint(self) -> str:
        r"""str: String describing any constraints."""
        out = ""
        if self.num_levels == 0:
            cond = []
            # TODO: Better handling of ndim > 1
            if any(self.bounds[0] != -np.inf):
                le = self.bounds[0][0] if self.ndim == 1 else self.bounds[0]
                cond.append(f"greater than {le}")
            if any(self.bounds[1] != np.inf):
                re = self.bounds[1][0] if self.ndim == 1 else self.bounds[1]
                cond.append(f"less than {re}")
            if len(cond) == 1:
                out = f"must be {cond[0]}"
            elif len(cond) == 2:
                out = f"must be {cond[0]} and {cond[1]}"
            return out
        elif self.num_levels == -1:
            pass
        else:
            out = (
                "must be one of "
                + self._format_choice_list(self.levels)
            )
        return out


class DoNothingModelAction(ModelAction):
    r"""Specific case of a model action to do nothing."""

    def __init__(
            self,
            name: Optional[str] = "donothing",
            description: Optional[str] = "Do nothing.",
            keywords: Optional[list] = [
                "do nothing", "take no action"
            ]):
        super().__init__(
            name,
            description=description,
            num_levels=-1,
            allow_donothing=False,
            keywords=keywords,
        )


class ModelActionSet(CachedPropertyMixin):
    r"""Set of model actions.

    Args:
        action_map: Mapping between action names and descriptions.
        num_levels: Number of levels per action if not specified in
            action_map (0 for continuous, -1 for boolean).
        allow_donothing: Include non-action as a possible action.
        exclusive: Don't allow more than one action per step.
        default_action_map: Mapping of default action descriptions that
            should be used to fill in missing information in action_map.
        param: Action parameters to use keyed to action names.

    """

    def __init__(
            self, action_map: dict,
            num_levels: Optional[int] = 0,
            allow_donothing: Optional[bool] = True,
            exclusive: Optional[bool] = True,
            default_action_map: Optional[dict] = None,
            param: Optional[dict] = None,
    ):
        self.actions = {}
        self.num_levels = num_levels
        self.allow_donothing = allow_donothing
        self.exclusive = exclusive
        if self.exclusive and self.allow_donothing:
            self.actions[""] = DoNothingModelAction()
        self.action_allow_donothing = (
            self.allow_donothing and (not self.exclusive)
        )
        default_action_map = default_action_map or {}
        param = param or {}
        for action, desc in action_map.items():
            if isinstance(desc, ModelAction):
                self.actions[action] = desc
                continue
            kws = desc.copy()
            kws.setdefault("num_levels", self.num_levels)
            if action in default_action_map:
                for k, v in default_action_map[action].items():
                    kws.setdefault(k, v)
            if action in param:
                kws["param"] = param["action"]
            self.actions[action] = ModelAction(
                action, allow_donothing=self.action_allow_donothing,
                **kws)
        super().__init__()
        self._on_edit_actions(in_init=True)

    @classmethod
    def create(cls, action_map: Union["ModelActionSet", dict],
               **kwargs) -> "ModelActionSet":
        r"""Create a ModelActionSet from a dictionary. If an existing
        ModelActionSet instance is provided, it is returned.

        Args:
            action_map: Existing ModelActionSet or dictionary.
            **kwargs: Additional keyword arguments are passed to the
                constructor if action_map is not a ModelActionSet
                instance.

        Returns:
            ModelActionSet instance.

        """
        if isinstance(action_map, ModelActionSet):
            # for k, v in kwargs.items():
            #     assert getattr(action_map, k, None) == v
            return action_map
        return cls(action_map, **kwargs)

    def __getitem__(self, k):
        return self.actions[k]

    def __contains__(self, k):
        return k in self.actions

    def items(self):
        r"""Action items."""
        return self.actions.items()

    def keys(self):
        r"""Action keys."""
        return self.actions.keys()

    def values(self):
        r"""Action values."""
        return self.actions.values()

    def pop(self, k, default=NoDefault):
        r"""Remove an action."""
        if k in self.actions:
            out = self.actions.pop(k)
            self._on_edit_actions()
            return out
        if default is NoDefault:
            raise KeyError(k)
        return default

    def _on_edit_actions(self, in_init=False):
        if self.exclusive and self.discrete:
            nprev = 0
            for desc in self.actions.values():
                desc.offset = nprev
                nprev += desc.num_choices
        if in_init:
            return
        self._clear_cached_properties()

    def set_param(self, param: dict, action: Optional[str] = None,
                  src: Optional[str] = "set_param"):
        r"""Update the action parameters that match the provided keywords.

        Args:
            param: Action parameters.
            action: Name of the action that should be updated. If not
                provided, all actions with matching parameters will be
                updated.
            src: Description of how the parameter is being updated for
                logging parameter conflicts.

        """
        if not param:
            return
        if action:
            self.actions[action].set_param(param, src=src)
            self._on_edit_actions()
            return
        update = False
        for action, desc in self.actions.items():
            iparam = {
                k: v for k, v in param.items()
                if k in desc.param_desc and k != desc.action_param
            }
            if iparam:
                update = True
                desc.set_param(iparam, src=src)
        if update:
            self._on_edit_actions()

    def scale_action_amounts(self, scale: Union[int, float]):
        r"""Scale action limits/levels.

        Args:
            scale: Amount to scale values by.

        """
        updated = False
        for v in self.actions.values():
            if not v.numeric:
                continue
            v.scale_action_amounts(scale)
            updated = True
        if updated:
            self._on_edit_actions()

    @readonly_cached_property
    def action_order(self) -> list:
        r"""list: Order of actions for indexing."""
        if self.exclusive and self.discrete:
            out = []
            for name, desc in self.actions.items():
                out += desc.num_choices * [name]
            return out
        return list(self.actions.keys())

    @readonly_cached_property
    def num_choices(self) -> int:
        r"""int: Number of discrete choices allowed for this action."""
        return sum(x.num_choices for x in self.actions.values())

    @readonly_cached_property
    def ndim(self) -> int:
        r"""int: Number of dimensions."""
        return sum(
            1 if x.discrete else x.ndim
            for x in self.actions.values()
        )

    @readonly_cached_property
    def discrete(self) -> bool:
        r"""bool: True if the action is discrete."""
        return all(x.discrete for x in self.actions.values())

    @readonly_cached_property
    def donothin_action(self):
        r"""object: Action ID to do nothing."""
        if not self.allow_donothing:
            return None
        if not self.exclusive:
            return {}
        if self.discrete:
            return 0
        return (0, 0)

    @readonly_cached_property
    def example_value(self):
        r"""object: Example action value."""
        if self.exclusive:
            last = self.actions[self.action_order[-1]]
            return {last.name: last.example_value}
        return {
            k: v.example_value for k, v in self.actions.items()
        }

    @readonly_cached_property
    def example_args(self) -> dict:
        r"""dict: Example action args."""
        return self.value2args(self.example_value)

    @readonly_cached_property
    def example_description(self) -> str:
        r"""str: Example action description."""
        return self.format_description(value=self.example_value)

    @readonly_cached_property
    def description_lines(self) -> list:
        r"""list: Lines describing the action set."""
        return self.format_description(return_lines=True)

    @readonly_cached_property
    def description(self) -> str:
        r"""str: Description of the action set."""
        return "\n".join(self.description_lines)

    @readonly_cached_property
    def space(self) -> gym.spaces.space.Space:
        r"""gym.spaces.space.Space: Action space."""
        if self.exclusive:
            if self.discrete:
                return gym.spaces.Discrete(self.num_choices)
            else:
                return gym.spaces.OneOf([
                    desc.space for desc in self.actions.values()
                ])
        return gym.spaces.Dict({
            action: desc.space
            for action, desc in self.actions.items()
        })

    def description2action(self, description: str):
        r"""Parse a description to get an action ID.

        Args:
            description: Action description.

        Returns:
            object: Action ID.

        """
        return self.value2action(self.description2value(description))

    def description2value(self, description: str):
        r"""Parse a description for a action value.

        Args:
            description: Action description.

        Returns:
            dict: Action value map.

        """
        out = {}
        for k, v in self.actions.items():
            match = v.search_description(description)
            if not match:
                continue
            out[k] = v.match2value(match)
            if self.exclusive:
                return out
            description = (
                description[:match.start(0)] + description[match.end(0):]
            ).strip()
            if not description:
                break
        if (not description) and (not self.exclusive):
            return out
        parts = (
            [description] if self.exclusive
            else [x + '.' for x in description.split('.')]
        )
        matches = [{} for _ in parts]
        for i, description in enumerate(parts):
            for k, v in self.actions.items():
                if k in out:
                    continue
                try:
                    matches[i][k] = v.fuzzy_search_description(
                        description)
                except InvalidActionError:
                    continue
        errors = []
        for k in self.actions.keys():
            idx = [i for i, match in enumerate(matches)
                   if k in match]
            if len(idx) > 1:
                errors.append(
                    f"Action \"{k}\" matched more than one part of "
                    f"the description:\n    "
                    + "\n    ".join(f"\"{parts[i]}\"" for i in idx)
                )
        for i, match in enumerate(matches):
            if len(match) == 0:
                errors.append(
                    f"No action matches this part of the description "
                    f"\"{parts[i]}\""
                )
            elif len(match) == 1:
                out.update(match)
            else:
                errors.append(
                    f"More than one action {list(match.keys())} "
                    f"matches the same part of the description: "
                    f"\"{parts[i]}\""
                )
        if errors:
            raise InvalidActionError(
                "Failed to parse description:\n  - "
                + "\n  - ".join(errors))
        return out

    def value2action(self, value: dict):
        r"""Convert an action value into an action ID.

        Args:
            value: Action value map.

        Returns:
            int, tuple, dict: Action ID.

        """
        action = {
            k: self.actions[k].value2action(v)
            for k, v in value.items()
        }
        if not self.exclusive:
            return action
        assert len(value) == 1
        name, action = list(action.items())[0]
        if self.discrete:
            return action
        else:
            return (self.action_order.index(name), action)

    def action2value(self, action: Union[int, tuple, dict, np.ndarray]):
        r"""Convert an action ID into a map of action values.

        Args:
            action: Action ID.

        Returns:
            dict: Action value map.

        """
        if self.exclusive:
            if self.discrete:
                if not isinstance(action, (int, np.integer)):
                    raise InvalidActionError(
                        f"An exclusive, discrete set of actions "
                        f"requires an integer action ID, not "
                        f"{type(action)}"
                    )
                name = self.action_order[action]
                action = {name: action}
            else:
                if not isinstance(action, tuple):
                    raise InvalidActionError(
                        f"An exclusive, mixed (discrete & continous) "
                        f"set of actions requires a tuple action ID, "
                        f"not {type(action)}"
                    )
                name = self.action_order[action[0]]
                action = {name: action[1]}
        if isinstance(action, np.ndarray):
            if len(action) != self.ndim:
                raise InvalidActionError(
                    f"A non-exclusive set of actions requires a "
                    f"np.ndarray action ID with {self.ndim} "
                    f"elements, not {len(action)}"
                )
            out = {}
            pos = 0
            for k, v in self.action.items():
                if v.discrete:
                    x = action[pos]
                    pos += 1
                else:
                    x = action[pos:(pos + v.ndim)]
                    pos += v.ndim
                out[k] = v.action2value(x)
            return out
        if not isinstance(action, dict):
            raise InvalidActionError(
                f"A non-exclusive set of actions requires a dict "
                f"action ID, not {type(action)}"
            )
        return {
            k: self.actions[k].action2value(v)
            for k, v in action.items()
        }

    def value2args(self, value: dict) -> Dict[str, tuple]:
        r"""Convert an action value map to an argument map.

        Args:
            value: Action value.

        Returns:
            dict: Action argument map.

        """
        out = {}
        for k, v in value.items():
            if self.actions[k].boolean and v is False:
                continue
            if isinstance(self.actions[k], DoNothingModelAction):
                continue
            out[k] = self.actions[k].value2args(v)
        return out

    def action2description(
            self, action: Union[int, tuple, dict, np.ndarray]) -> str:
        r"""Convert an action ID into a natural language description.

        Args:
            action: Action ID.

        Returns:
            str: Action description.

        """
        return self.format_description(value=self.action2value(action))

    def action2args(
            self, action: Union[int, tuple, dict, np.ndarray]) -> dict:
        r"""Convert an action ID into a parameter argument map for
        actvars.

        Args:
            action: Action ID.

        Returns:
            dict: Parameter to argument map.

        """
        return self.value2args(self.action2value(action))

    def format_description(
            self, value: Optional[dict] = None,
            return_lines: Optional[bool] = False,
    ) -> str:
        r"""Format a description of the action.

        Args:
            value: Action value to include in the description.
            return_lines: If True, return a list of lines instead of a
                merged string.

        Returns:
            str: Formatted action description. A list will be returned
                if return_lines is True.

        """
        lines = []
        if value is None:
            if self.exclusive:
                lines.append(
                    "Available actions (pick exactly one):")
            else:
                lines.append(
                    "Available actions (include instructions for each):")
            lines += [
                "- " + v.format_description()
                for v in self.actions.values()
            ]
        else:
            if self.exclusive:
                assert len(value) == 1
            lines += [
                self.actions[k].format_description(value=v)
                for k, v in value.items()
            ]
            lines = [" ".join(lines)]  # Multiple actions
        return lines if return_lines else "\n".join(lines)


class BaseModelFile(CachedPropertyMixin, ABC):
    r"""Base class for managing model input files.

    Args:
        fname: Path to a model file.
        generated: If True, this file was generated.
        contents: Contents to initialize the file with.
        fname_orig: Original model file that this one was generated from.

    """

    CACHED = False

    def __init__(self, fname: str, generated: Optional[bool] = False,
                 contents: Optional[dict] = None,
                 fname_orig: Optional[str] = None):
        self.fname = fname
        self.fname_orig = fname_orig or fname
        self.generated = generated
        if contents:
            self.contents = contents
        super().__init__()

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        r"""Cleanup any generated file."""
        if self.generated and self.exists and not self.CACHED:
            os.remove(self.fname)
            self.generated = False
            self._clear_cached_properties()

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

    def _get(self, name: str):
        r"""Get a parameter from the model file.

        Args:
            name: Parameter name.

        Returns:
            Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        raise KeyError(name)

    def _set(self, name: str, value: Any) -> Any:
        r"""Set a parameter in the model file.

        Args:
            name: Parameter name.
            value: Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        raise KeyError(name)

    @staticmethod
    def parameter_property(method: Callable):
        r"""Decorator for a BaseModelFile method that produces the default
        value that should be used if a KeyError is not raised by
        BaseModelFile.get(<property name>).

        Args:
            method: BaseModelFile method being wrapped.

        """

        name = method.__qualname__.rsplit('.', 1)[-1]

        @property
        def _parameter_property(self):
            try:
                return self.get(name)
            except KeyError:
                self._cached_properties[name] = method(self)
                return self._cached_properties[name]

        return _parameter_property

    @cached_property
    def contents(self):
        r"""object: File contents."""
        return self._read(self.fname)

    @readonly_cached_property
    @abstractmethod
    def is_interactive(self):
        r"""bool: True if the model file is interactive."""
        raise NotImplementedError  # pragma: no cover

    @contextlib.contextmanager
    def prevent_overwrite(self, suffix: Optional[str] = "-Modified"):
        r"""Context to ensure that a duplicate is made if the context
        exits successfully during modification of the file contents.

        Args:
            suffix: File suffix to add if a new file name is generated.

        """
        assert self.contents  # Ensure contents loaded
        yield
        if self.exists:
            self.move(suffix=suffix)
        self.generated = False
        self._clear_cached_properties()

    @parameter_property
    def output_vars(self):
        r"""list: Output variables."""
        return []

    @property
    def exists(self):
        r"""bool: True if the file exists."""
        return os.path.isfile(self.fname)

    def get(self, name: str, default=NoDefault):
        r"""Get a parameter from the model file.

        Args:
            name: Parameter name.
            default: Value to return if the parameter can't be found.

        Returns:
            Parameter value.

        """
        if name in self._cached_properties:
            return self._cached_properties[name]
        try:
            out = self._get(name)
            self._cached_properties[name] = out
            return out
        except KeyError:
            if default is not NoDefault:
                return default
            raise

    def set(self, name: str, value: Any) -> Any:
        r"""Set a parameter in the model file.

        Args:
            name: Parameter name.
            value: Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        with self.prevent_overwrite():
            self._set(name, value)

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
            self._clear_cached_properties()
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
        if dst is None:
            if suffix:
                dst = suffix.join(os.path.splitext(self.fname))
            else:
                dst = self.fname
        if directory is not None:
            dst = os.path.join(directory, os.path.basename(dst))
        while os.path.isfile(dst):
            dst = str(uuid.uuid4()).join(os.path.splitext(dst))
        if dst != self.fname:
            self.generated = False
        self.fname = dst
        if self.exists:
            raise ValueError(f"Cannot move to a file that already exists: "
                             f"\"{self.fname}\"")
        return self.fname

    def copy(self, **kwargs) -> "BaseModelFile":
        r"""Create a copy of this .apsimx model.

        Args:
            **kwargs: Addiitonal keyword arguments are passed to move.

        Returns:
            ApsimXFile: Copied .apsimx model.

        """
        out = type(self)(self.fname, generated=self.generated,
                         fname_orig=self.fname_orig)
        out.contents = copy.deepcopy(self.contents)
        if kwargs:
            out.move(**kwargs)
        return out

    def make_interactive(self, actions: list):
        r"""Modify this file to make it interactive.

        Args:
            actions: List of actions that should be enabled.

        """
        if self.is_interactive:
            logger.warning(
                f"Source model file \"{self.fname}\" is already "
                f"interactive"
            )
            return
        with self.prevent_overwrite(suffix="-Interactive"):
            self._make_interactive(actions)

    @abstractmethod
    def _make_interactive(self, actions: list):
        r"""Modify this file to make it interactive.

        Args:
            actions: List of actions that should be enabled.

        """
        raise NotImplementedError  # pragma: no cover


class BaseModelEngine(ABC):
    r"""Base class for exposing a model as an environment engine.

    Args:
        model_file: Path to one or more model input files.
        model_dir: Path to the directory containing the model.
        model_suffix: Additional suffix to add to a copy of the provided
            model file to ensure that it is unique.
        output_dir: Path to the directory where output should be saved.
        start_time: Simulation start time.
        end_time: Simulation end time.
        duration: Simulation duration. Only used if either start_time or
            end_time is not provided.
        param: Model parameters to update at the beginning of the
            simulation.
        actions: Names of actions to include. Only used if action_map
            not provided.
        action_map: Description of actions available via the act method.
        action_param: Action parameters to use keyed to action names.

    """

    INPUT_FILE_TYPE = None
    AVAILABLE_ACTION_MAP = {}
    EXPLICIT_PARAM = ["start_time", "end_time", "duration"]
    DATE_PARAM = [("start_time", "end_time", "duration")]
    DEFAULT_PARAM = {}

    def __init__(
            self,
            model_file: Union[str, List[str]],
            model_dir: Optional[str] = None,
            model_suffix: Optional[str] = None,
            output_dir: Optional[str] = None,
            start_time: Optional[datetime.datetime] = None,
            end_time: Optional[datetime.datetime] = None,
            duration: Optional[datetime.timedelta] = None,
            param: Optional[dict] = None,
            actions: Optional[List[str]] = None,
            action_map: Optional[Union[dict, ModelActionSet]] = None,
            action_param: Optional[dict] = None,
    ):
        self.model_file = model_file
        self.model_dir = model_dir
        self.model_suffix = model_suffix
        self.output_dir = output_dir
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.initial_param = param.copy() if param is not None else {}
        self.initial_param_static = {}
        self.initial_param_dynamic = {}
        self.initial_param_src = {}
        self.history = defaultdict(lambda: [])
        if ((self.model_file and self.model_dir
             and (not os.path.isfile(self.model_file))
             and (not os.path.isabs(self.model_file))
             and os.path.isfile(
                 os.path.join(self.model_dir, self.model_file)))):
            self.model_file = os.path.join(
                self.model_dir, self.model_file)
        self.model = self.INPUT_FILE_TYPE(self.model_file)
        self.action_map = ModelActionSet.create(
            action_map or self.select_actions(actions),
        )
        if action_param:
            for k, v in action_param.items():
                self.action_map.set_param(v, action=k)
        self.update_model_file()

    def has_param(self, name: str,
                  skip_file: Optional[bool] = False) -> bool:
        r"""Check if a model has a parameter value set.

        Args:
            name: Name of parameter to check.
            skip_file: If True, don't try to read the parameter from the
                file.

        Returns:
            bool: True if the parameter is set, False otherwise.

        """
        try:
            self.get_param(name, skip_file=skip_file)
            return True
        except KeyError:
            return False

    def del_param(self, name: str, src: Optional[List[str]] = None) -> bool:
        r"""Clear a model parameter.

        Args:
            name: Name of parameter to clear.
            src: If provided, only delete the parameter if the source is
                 one of the listed values.

        """
        if src and not self.initial_param_src.get(name, "").startswith(src):
            return
        self.initial_param.pop(name, None)
        self.initial_param_src.pop(name, None)
        self.initial_param_static.pop(name, None)
        self.initial_param_dynamic.pop(name, None)
        if name in self.EXPLICIT_PARAM:
            setattr(self, name, None)

    def set_param(self, name: str, value: Any,
                  src: Optional[str] = "USER",
                  dont_update: Optional[bool] = False) -> bool:
        r"""Set a model parameter, updating the value for actions where
        appropriate.

        Args:
            name: Name of parameter to set.
            value: Parameter value.
            src: Description of where the parameter came from.
            dont_update: If True, don't update the model file.

        Returns:
            bool: True if the set was successful.

        """
        if self.is_running:
            raise KeyError(f"Cannot update initial parameter \"{name}\" "
                           f"after the model is running")
        self.initial_param[name] = value
        if name in self.EXPLICIT_PARAM:
            setattr(self, name, value)
        self.initial_param_src[name] = src
        if self.initial_param_static.get(name, None) != value:
            self.initial_param_static.pop(name, None)
        if self.initial_param_dynamic.get(name, None) != value:
            self.initial_param_dynamic.pop(name, None)
        other_values = {}
        names_update = [name]
        if name not in self.initial_param_static:
            if name == "year":
                for k1, k2, kdiff in self.DATE_PARAM:
                    v1 = self.get_param(k1, None, skip_file=True)
                    v2 = self.get_param(k2, None, skip_file=True)
                    if v1 is not None and v2 is not None:
                        vdiff = v2 - v1
                    else:
                        vdiff = self.get_param(kdiff, None,
                                               skip_file=True)
                    if v1 is not None:
                        other_values[k1] = (v1, v1.replace(year=value))
                        if vdiff is not None:
                            other_values[k2] = (
                                v2, other_values[k1][1] + vdiff)
                    elif v2 is not None:
                        other_values[k2] = (v2, v2.replace(year=value))
                        if vdiff is not None:
                            other_values[k1] = (
                                v1, other_values[k2][1] - vdiff)
            else:
                for x in self.DATE_PARAM:
                    if name not in x:
                        continue
                    for k in list(x) + ["year"]:
                        if k == name:
                            continue
                        if self.initial_param_src.get(
                                k, "").startswith("CALC"):
                            v = self.get_param(k, None, skip_file=True)
                            other_values[k] = (
                                v,
                                self.calc_param(k, None, skip_file=True)
                            )
        for k, (v, vnew) in other_values.items():
            if vnew is None or v == vnew:
                continue
            self.set_param(
                k, vnew,
                src=f"{self.initial_param_src[k]}-{name}",
                dont_update=True,
            )
            names_update.append(k)
        if dont_update:
            return True
        return self.update_param_in_file(names_update)

    def get_param(self, name: str, default: Optional[Any] = NoDefault,
                  skip_file: Optional[bool] = False,
                  skip_calc: Optional[bool] = False,
                  skip_default: Optional[bool] = False,
                  skip_src: Optional[List[str]] = None) -> Any:
        r"""Get a model parameter.

        Args:
            name: Name of parameter to get.
            default: Default to return if the parameter cannot be
                located.
            skip_file: If True, don't try to read the parameter from the
                file.
            skip_calc: If True, don't try to calculate missing parameters.
            skip_default: If True, don't use values in DEFAULT_PARAM for
                missing parameters.
            skip_src: Set of sources that should be ignored.

        Returns:
            Parameter value.

        Raises:
            KeyError: If a parameter value cannot be located and default
                is not provided.

        """
        if skip_src:
            if self.initial_param_src.get(name, "").startswith(skip_src):
                if default is not NoDefault:
                    return default
                raise KeyError(name)
            if "CALC" in skip_src:
                skip_calc = True
            if "FILE" in skip_src:
                skip_file = True
            if "DEFAULT" in skip_src:
                skip_default = True
        if self.EXPLICIT_PARAM:
            v = getattr(self, name, None)
            if v is not None:
                self.initial_param_src.setdefault(name, "ATTR")
                if ((name in self.initial_param
                     and self.initial_param[name] != v)):
                    logger.warning(
                        f"Parameter \"{name}\" specified as both an "
                        f"explicit keyword argument to {type(self)} "
                        f"and in \"param\". The keyword argument "
                        f"{v} will be used and the param value "
                        f"{self.initial_param[name]} will be discarded."
                    )
                return v
        if name in self.initial_param:
            self.initial_param_src.setdefault(name, "INIT")
            return self.initial_param[name]
        if not skip_calc:
            try:
                out = self.calc_param(name, skip_file=skip_file)
                self.initial_param_src[name] = "CALC"
                return out
            except KeyError:
                pass
        if (not skip_default) and name in self.DEFAULT_PARAM:
            out = self.DEFAULT_PARAM[name]
            self.initial_param_src[name] = "DEFAULT"
            return out
        if not skip_file:
            try:
                out = self.model.get(name)
                self.initial_param_src[name] = "FILE"
                return out
            except KeyError:
                pass
        if default is not NoDefault:
            return default
        raise KeyError(name)

    def calc_param(self, name: str, default: Optional[Any] = NoDefault,
                   **kwargs):
        r"""Calculate a parameter from other parameters.

        Args:
            name: Parameter to calculate.
            default: Default to return if the parameter cannot be
                calculated.
            **kwargs: Additional keyword arguments are passed to
                get_param when getting parameters used in the
                calculation.

        Returns:
            Calculated parameter value.

        """
        # Prevent infinite recursion
        kws = dict(kwargs, skip_calc=True)
        for k1, k2, kdiff in self.DATE_PARAM:
            if name not in [k1, k2, kdiff]:
                continue
            try:
                if name == k1:
                    v2 = self.get_param(k2, **kws)
                    vdiff = self.get_param(kdiff, **kws)
                    out = v2 - vdiff
                elif name == k2:
                    v1 = self.get_param(k1, **kws)
                    vdiff = self.get_param(kdiff, **kws)
                    out = v1 + vdiff
                elif name == kdiff:
                    v1 = self.get_param(k1, **kws)
                    v2 = self.get_param(k2, **kws)
                    out = v2 - v1
                return out
            except KeyError:
                pass
        if default is not NoDefault:
            return default
        raise KeyError(name)

    def update_param_in_file(
            self, names: Optional[List[str]] = None,
            required: Optional[bool] = False,
    ) -> bool:
        r"""Update a parameter in the model file if it has changed.

        Args:
            names: Names of parameters to set. If not provided, all
                parameters that have been updated since the last time
                update_param_in_file was called will be set.
            required: If True, a KeyError will be raised if a value
                cannot be updated for any of the specified names.

        Returns:
            bool: True if the update was successful.

        """
        if names is None:
            names = list(self.initial_param.keys())
        if self.is_running:
            raise KeyError(f"Cannot update initial parameters "
                           f"\"{names}\" after the model is running")
        missing = []
        added = {}
        for k in names:
            if k not in self.initial_param:
                missing.append(k)
                continue
            elif k in self.initial_param_static:
                continue
            elif k in self.initial_param_dynamic:
                missing.append(k)
                continue
            value = self.initial_param[k]
            added[k] = value
            try:
                if self.initial_param_src[k] != "FILE":
                    self.model.set(k, value)
                self.initial_param_static[k] = value
            except KeyError:
                self.initial_param_dynamic[k] = value
                missing.append(k)
        if added:
            self.action_map.set_param(added, src="model parameters")
            logger.info(f"Synchronized parameters:\n"
                        f"{pprint.pformat(added)}")
        if required and missing:
            raise KeyError(f"Failed to update parameters: "
                           f"{missing}")
        logger.info(
            f"initial_param:\n{pprint.pformat(self.initial_param)}")

    def sync_param(self, names: Optional[List[str]] = None,
                   required: Optional[bool] = False,
                   dont_update: Optional[bool] = False,
                   skip_file: Optional[bool] = False, **kwargs):
        r"""Set/get explicit model file parameters.

        Args:
            names: Names of parameters to synchronize.
            required: If True, a KeyError will be raised if a value
                cannot be found for any of the specified names.
            dont_update: If True, don't update the model file.
            skip_file: If True, only sync parameters between initial_param
                and attributes for EXPLICIT_PARAM, but do not inspect
                the file.
            **kwargs: Additional keyword arguments are passed to
                get_param for each name.

        Raises:
            KeyError: If required is True and a value cannot be found
                for any of the specified names.

        """
        if names is None:
            names = list(self.initial_param.keys())
            if not skip_file:
                names = list(set(names) | set(self.EXPLICIT_PARAM))
        elif isinstance(names, str):
            names = [names]
        for k in names:
            try:
                v = self.get_param(k, skip_file=skip_file, **kwargs)
            except KeyError:
                continue
            self.set_param(k, v, src=self.initial_param_src[k],
                           dont_update=True)
        if not dont_update:
            self.update_param_in_file(names, required=required)

    def update_model_file(self):
        r"""Update the model file to make it interactive and set the
        start/end times."""
        if not self.model.is_interactive:
            self.model.make_interactive(list(self.action_map.keys()))
        if self.output_dir or self.model_suffix:
            self.model.move(directory=self.output_dir,
                            suffix=self.model_suffix)
        self.sync_param()
        for k1, k2, kdiff in self.DATE_PARAM:
            v1 = getattr(self, k1, None)
            v2 = getattr(self, k2, None)
            if v1 is not None and v2 is not None and v1 >= v2:
                raise ValueError(f"{k2} ({v2}) does "
                                 f"not come after {k1} "
                                 f"({v1})")
        if not self.model.exists:
            self.model.write()

    @classmethod
    def select_actions(cls, actions: Optional[List[str]] = None,
                       action_map: Optional[dict] = None):
        r"""Select a set of default actions.

        Args:
            actions: Set of actions to select.
            action_map: Map that actions should be selected from.

        Returns:
            dict: Description of selected actions.

        """
        action_map = action_map or {}
        actions = (actions or list(action_map.keys())
                   or list(cls.AVAILABLE_ACTION_MAP.keys()))
        return {
            k: action_map.get(k, cls.AVAILABLE_ACTION_MAP[k].copy())
            for k in actions
        }

    def get_output_vars(self) -> List[str]:
        r"""Get the output variables specified by the model file.

        Returns:
            list: Output variables

        """
        return self.model.output_vars

    @property
    def is_complete(self):
        r"""bool: True if the simulation is complete."""
        return self.current_time >= self.end_time

    def __del__(self):
        self.stop(cleanup=True)

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

    def start(self):
        r"""Start the model engine."""
        self._start()
        self.setvars(self.initial_param_dynamic)
        logger.info(f"Simulating from {self.start_time} to {self.end_time}")
        # added = {}
        # for k in self.EXPLICIT_PARAM:
        #     v = getattr(self, k, None)
        #     if v is not None:
        #         continue
        #     setattr(self, k, self.get(k, allow_error=True))
        #     if getattr(self, k) is not None:
        #         added[k] = getattr(self, k)
        # if added:
        #     logger.info(f"Updated attributes from running model:\n"
        #                 f"{pprint.pformat(added)}")

    @abstractmethod
    def _start(self):
        r"""Start the model engine."""
        raise NotImplementedError  # pragma: no cover

    def stop(self, cleanup: Optional[bool] = False):
        r"""Stop the model engine.

        Args:
            cleanup: If True, cleanup the generated model file.

        """
        try:
            self._stop()
        finally:
            if cleanup:
                self.model.cleanup()

    @abstractmethod
    def _stop(self):
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
            kws = {}
            if action in self.action_map:
                kws = self.action_map[action].combine_args_and_kwargs(
                    args, kwargs)
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
        while ((self.is_running and self.current_time < time
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
        if self.current_time < time:
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


class BaseModelLLMPromptGenerator(CachedPropertyMixin, ABC):
    """Generate LLM prompts for environments.

    This class handles the creation of system prompts and turn prompts
    for LLM-based agricultural management agents.
    """

    VALID_THINKING_MODES = {"minimal", "grounding_decision"}
    THINKING_MODE_ALIASES = {
        "think": "grounding_decision",
    }
    DEFAULT_REWARD = ""
    DEFAULT_STATE_DESCRIPTOR = ""
    DEFAULT_DESC_MAP = {}

    def __init__(
            self,
            num_levels: Optional[int] = 4,
            intervention_interval: Optional[int] = 7,
            output_vars: Optional[List[str]] = None,
            desc_map: Optional[Dict[str, Tuple[str, str]]] = None,
            action_map: Optional[Union[dict, ModelActionSet]] = None,
            reward: Optional[str] = None,
            state_descriptor: Optional[str] = None,
            allow_donothing: Optional[bool] = True,
            exclusive: Optional[bool] = True,
            require_think: bool = False,
            thinking_mode: str = "grounding_decision",
            think_tag: str = "think",
            answer_tag: str = "answer",
    ):
        """Initialize the prompt generator.

        Args:
            num_levels: Number of levels per action if not specified in
                action_map (0 for continuous).
            intervention_interval: Days between decisions
            output_vars: List of observation variable names
            desc_map: Custom description mapping for variables
            action_map: Custom description mapping for actions
            reward: Description of the reward that should be used.
            state_descriptor: Description of the overall type of state
                information.
            allow_donothing: Include non-action as a possible action.
            exclusive: Don't allow more than one action per step.
            require_think: Whether to require thinking before answering
            thinking_mode: Thinking prompt variant when require_think=True.
                Supported values: "minimal", "think" (alias of
                "grounding_decision"), "grounding_decision"
            think_tag: Tag name for thinking (default: "think", e.g.
                "tool_call")
            answer_tag: Tag name for answer (default: "answer")

        """
        self.num_levels = num_levels
        self.intervention_interval = intervention_interval
        self.output_vars = output_vars or []
        self.desc_map = desc_map or self.DEFAULT_DESC_MAP
        self.reward = reward or self.DEFAULT_REWARD
        self.state_descriptor = (
            state_descriptor or self.DEFAULT_STATE_DESCRIPTOR)
        if self.state_descriptor:
            self.state_descriptor = self.state_descriptor.strip() + " "
        self.allow_donothing = allow_donothing
        self.exclusive = exclusive
        self.require_think = require_think
        self.thinking_mode = self._normalize_thinking_mode(thinking_mode)
        self.think_tag = think_tag
        self.answer_tag = answer_tag
        self.action_map = ModelActionSet.create(
            action_map or {},
            num_levels=self.num_levels,
            allow_donothing=self.allow_donothing,
            exclusive=self.exclusive,
        )
        super().__init__()

    @readonly_cached_property
    def reward_inline(self) -> str:
        r"""str: Version of the reward for inclusion in statements."""
        return self.reward.lower().strip(".")

    @readonly_cached_property
    def state_grounding(self) -> str:
        r"""str: State grounding description."""
        return (
            f"State grounding: describe the current "
            f"{self.state_descriptor}state, the main limiting "
            f"factor and any missing information based on the "
            f"current observations"
        )

    @abstractmethod
    def turn_context(self, observation: np.ndarray) -> str:
        r"""Generate a string to summarize the current turn at a high
        level with the context of the simulation.

        Args:
            observation: Current state.

        Returns:
            str: Turn summary.

        """
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def _normalize_thinking_mode(cls, thinking_mode: str) -> str:
        """Normalize and validate the thinking mode name."""
        normalized = str(thinking_mode).strip().lower().replace("-", "_")
        normalized = cls.THINKING_MODE_ALIASES.get(normalized, normalized)
        if normalized not in cls.VALID_THINKING_MODES:
            valid = ", ".join(
                sorted(cls.VALID_THINKING_MODES
                       | set(cls.THINKING_MODE_ALIASES)))
            raise ValueError(
                f"Invalid thinking_mode '{thinking_mode}'. Expected one "
                f"of: {valid}")
        return normalized

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Generate the system prompt for the LLM agent.

        Returns:
            System prompt string

        """
        raise NotImplementedError  # pragma: no cover

    def get_turn_prompt(self, observation: np.ndarray) -> str:
        """Generate the complete per-turn user prompt.

        Supports plain-answer mode plus two thinking guidance variants.
        Structure: intro → observation → actions → guidance → format.

        """
        output_vars = self.output_vars
        if not output_vars:
            raise ValueError("output_vars must be provided in __init__")

        # ── 1. Intro ────────────────────────────────────────────────
        intro = self.turn_context(observation)

        bridge = (
            "Below is the current observation for this step."
        )

        # ── 2. Observation block ────────────────────────────────────
        obs_sections = {}
        for key, value in zip(output_vars, observation):
            section, desc = self.desc_map.get(key, ("Other", key))
            if section == "Timeline":
                continue
            obs_sections.setdefault(section, [])
            obs_sections[section].append(f"- {desc}: {value:.4g}")
        obs_lines = []
        for section, section_lines in obs_sections.items():
            obs_lines.append(f"[{section}]")
            obs_lines += section_lines

        # ── 3. Action options ───────────────────────────────────────
        action_lines = self.action_map.description_lines.copy()
        example_action = self.action_map.example_description

        # ── 4. Decision guidance (varies by require_think × thinking_mode)
        if not self.require_think:
            guidance_intro = (
                "Please consider the following when making a decision:"
            )
            action_lines.extend([
                "",
                guidance_intro,
                f"1. {self.state_grounding}",
                f"2. Decision: determine which available action is most "
                f"likely to {self.reward_inline} at this step",
            ])
        elif self.thinking_mode == "grounding_decision":
            action_lines.extend([
                "",
                "Please reason briefly before action about:",
                f"1. {self.state_grounding}",
                f"2. Decision: determine which available action is most "
                f"likely to {self.reward_inline} at this step",
            ])
        else:
            action_lines.extend([
                "",
                "Please think about your choice before answering.",
            ])

        # ── 5. Response format (varies by require_think) ───────────
        at = self.answer_tag
        if self.require_think:
            tt = self.think_tag
            if self.thinking_mode == "grounding_decision":
                action_lines.extend([
                    "",
                    "Keep the reasoning concise and decision-focused. "
                    "Do not restate the full input.",
                    "",
                    f"Respond using the exact format: <{tt}> ... </{tt}> "
                    f"<{at}> ... </{at}> with no extra text.",
                    "",
                    f"Example: <{tt}>[reasoning content]</{tt}> "
                    f"<{at}>{example_action}</{at}>",
                ])
            else:
                action_lines.extend([
                    "",
                    f"Respond using the exact format: <{tt}> ... </{tt}> "
                    f"<{at}> ... </{at}> with no extra text.",
                    "",
                    f"Example: <{tt}>[reasoning content]</{tt}> "
                    f"<{at}>{example_action}</{at}>",
                ])
        else:
            action_lines.extend([
                "",
                f"Respond using the exact format: <{at}> ... </{at}> "
                f"with no extra text.",
                "",
                f"Example: <{at}>{example_action}</{at}>",
            ])

        # ── Assemble (sections joined by blank lines) ───────────────
        sections = [intro, bridge]
        sections.append(
            f"<current observation>\n{chr(10).join(obs_lines)}\n"
            f"</current observation>"
        )
        sections.append("\n".join(action_lines))
        return "\n\n".join(sections)

    def describe_action(self, action_id: Union[int, dict, tuple]) -> str:
        """Convert action ID to natural language description.

        Args:
            action_id: Integer action ID from the environment

        Returns:
            Natural language description in <answer>...</answer> format

        """
        at = self.answer_tag
        try:
            action = self.action_map.action2description(action_id)
            return f"<{at}>{action}</{at}>"
        except BaseException:
            return f"<{at}>Unknown action {action_id}.</{at}>"

    def parse_action_response(
            self, response: str) -> Optional[Union[int, dict, tuple]]:
        """Parse LLM response to extract action ID.

        Both modes use strict fullmatch — any extra content is invalid.
        Model-inherent thinking is extracted by the model interface layer
        before the response reaches this method.

        - ``require_think=True``:  ``<tag>...</tag><answer>...</answer>``
        - ``require_think=False``: ``<answer>...</answer>`` only

        """
        at = re.escape(self.answer_tag)
        if self.require_think:
            tt = re.escape(self.think_tag)
            m = re.fullmatch(
                rf"\s*<{tt}>(.*?)</{tt}>\s*<{at}>(.*?)</{at}>\s*",
                response,
                re.DOTALL,
            )
        else:
            m = re.fullmatch(
                rf"\s*<{at}>(.*?)</{at}>\s*",
                response,
                re.DOTALL,
            )
        if m is None:
            return None
        action_text = m.group(m.lastindex).strip()
        try:
            return self.action_map.description2action(action_text)
        except InvalidActionError:
            return None

    @classmethod
    def from_env(
            cls,
            env: "BaseModelEnv",
            **kwargs
    ) -> "BaseModelLLMPromptGenerator":
        """Create prompt generator from a model gym environment.

        Args:
            env: model gym environment instance
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            BaseModelLLMPromptGenerator instance configured for the
                environment.

        """
        return cls(
            num_levels=env.num_levels,
            intervention_interval=env.intervention_interval,
            output_vars=env.output_vars,
            action_map=env.action_map,
            allow_donothing=env.allow_donothing,
            exclusive=env.exclusive,
            # REWARD
            **kwargs
        )


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
            action_map (0 for continuous, -1 for boolean).
        actions: Names of actions to include. Only used if action_map
            not provided.
        action_map: Custom description mapping for actions.
        revenue_var: Description of how profit should be calculated from
            an output variable.
        model_param: Initial model parameters to set in the model file
            and/or when the simulation begins.
        action_param: Action parameters to use keyed to action names.
        allow_donothing: Include non-action as a possible action.
        exclusive: Don't allow more than one action per step.
        **kwargs: Additional keyword arguments are passed to the model
            engine constructor.

    """

    MODEL_ENGINE_CLASS = None
    LLM_PROMPT_GENERATOR_CLASS = None
    DEFAULT_ACTIONS = []
    DEFAULT_REVENUE_VAR = {}

    def __init__(
            self,
            model_file: Optional[str] = None,
            start_time: Optional[datetime.datetime] = None,
            end_time: Optional[datetime.datetime] = None,
            intervention_interval: Optional[
                Union[int, datetime.timedelta]] = 7,
            output_vars: Optional[List[str]] = None,
            num_levels: Optional[int] = 4,
            actions: Optional[List[str]] = None,
            action_map: Optional[Union[dict, ModelActionSet]] = None,
            revenue_var: Optional[Dict[str, Union[str, float]]] = None,
            model_param: Optional[dict] = None,
            action_param: Optional[dict] = None,
            allow_donothing: Optional[bool] = True,
            exclusive: Optional[bool] = True,
            scale_action_amounts_by_interval: Optional[bool] = False,
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
        self.revenue_var = revenue_var or copy.deepcopy(
            self.DEFAULT_REVENUE_VAR)
        self.model_param = model_param
        self.allow_donothing = allow_donothing
        self.exclusive = exclusive
        self.model_kwargs = kwargs
        action_map = action_map or {}
        actions = (
            actions or list(action_map.keys())
            or self.DEFAULT_ACTIONS
        )
        self.action_map = ModelActionSet.create(
            self.MODEL_ENGINE_CLASS.select_actions(actions, action_map),
            num_levels=self.num_levels,
            allow_donothing=self.allow_donothing,
            exclusive=self.exclusive,
            default_action_map=self.MODEL_ENGINE_CLASS.AVAILABLE_ACTION_MAP,
        )
        if action_param:
            for k, v in action_param.items():
                self.action_map.set_param(v, action=k)
        if scale_action_amounts_by_interval:
            self.action_map.scale_action_amounts(
                float(self.intervention_interval.days)
            )

        self.model = self.create_model(**self.model_kwargs)
        self.model.start()
        if self.start_time is None:
            self.start_time = self.model.start_time
        if self.end_time is None:
            self.end_time = self.model.end_time
        if not self.output_vars:
            self.output_vars = self.model.get_output_vars()
        if self.revenue_var:
            self.output_vars = [self.revenue_var["name"]] + [
                k for k in self.output_vars
                if k != self.revenue_var["name"]
            ]

        # Define what actions are available (bounds for each parameter)
        self.action_space = self.action_map.space

        # Define what the agent can observe (bounds for each output)
        self.observation_space = gym.spaces.Box(
            shape=(len(self.output_vars, ), ),
            low=-np.inf, high=np.inf,
            dtype=np.float64,
        )

        self.log = self._init_log()

    @property
    def current_time(self):
        r"""datetime.dateime: Current time."""
        return self.model.current_time

    @property
    def intervention_timedelta(self):
        r"""datetime.timedelta: Intervention interval as delta."""
        if isinstance(self.intervention_interval, datetime.timedelta):
            return self.intervention_interval
        return datetime.timedelta(self.intervention_interval)

    def get_output_vars(self) -> List[str]:
        r"""Get the output variables specified by the model file.

        Returns:
            list: Output variables

        """
        return self.output_vars

    def _init_log(self) -> dict:
        r"""Initialize the log."""
        return {
            "action": dict(),
            "obs": dict(),
            "cost": dict(),
            "reward": dict(),
            "time": dict(),
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
        self.log["time"][self.current_time] = self.current_time

    def create_model(self, **kwargs) -> BaseModelEngine:
        r"""Create a new model engine."""
        return self.MODEL_ENGINE_CLASS(
            model_file=self.model_file,
            start_time=self.start_time,
            end_time=self.end_time,
            action_map=self.action_map,
            param=self.model_param,
            **kwargs
        )

    def close(self):
        r"""Close the environment."""
        self.model.stop(cleanup=True)
        super().close()

    def get_llm_prompt_generator(
            self, **kwargs) -> "BaseModelLLMPromptGenerator":
        r"""Create an LLM prompt generator for this environment.

        Args:
            **kwargs: Keyword arguments are passed to the from_env
                method of the LLM_PROMPT_GENERATOR_CLASS type.

        Returns:
            BaseModelLLMPromptGenerator: LLM prompt generator.

        """
        return self.LLM_PROMPT_GENERATOR_CLASS.from_env(self)

    def _get_cost(self, action: dict) -> float:
        r"""Calculate the cost of the current action.

        Args:
            action: Current action.

        Returns:
            float: Action cost.

        """
        out = 0.0
        for k, v in action.items():
            out += self.action_map[k].args2cost(v)
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
            obs[self.revenue_var["name"]]
            * self.revenue_var.get("cost", 1)
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
        cost = (
            sum(v for v in self.log["cost"].values())
            + self._get_cost(action)
        )
        return revenue - cost

    def _get_obs(self) -> np.ndarray:
        r"""Observe the current state of the model.

        Returns:
            dict: Output state properties.

        """
        return self.model.getvars(self.output_vars)

    def _process_observation(self, observation: dict) -> np.ndarray:
        r"""Force the observations into the expected format.

        Args:
            observation: Raw observations.

        Returns:
            np.ndarray: Array of observations parameters.

        """
        observation = np.array(list(observation.values()))
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
        action_dict = self.action_map.action2args(action)
        self.model.actvars(action_dict)
        self.model.fast_forward(self.intervention_timedelta)
        observation = self._get_obs()
        reward = self._get_reward(action_dict, observation)
        terminated = (not self.model.is_running)
        truncated = self.model.is_complete
        self._log(action_dict, observation, reward)
        return (self._process_observation(observation), reward,
                terminated, truncated, self.log)
