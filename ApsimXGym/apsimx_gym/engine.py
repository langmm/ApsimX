import os
import re
import json
import copy
import zmq
import msgpack
import subprocess
import contextlib
import datetime
import numpy as np
import pandas as pd
from typing import Optional, Union, Any, List, Callable
from . import logger
from .utils import _gymdir, _apsimxdir, LogPipe
from .base import (
    readonly_cached_property,
    RecoverableError, ModelEngineError, InvalidActionError,
    RecoverableModelEngineError,
)
from .crop import (
    CropModelFile, BaseWeatherFile, CropModelEngine,
    CropModelLLMPromptGenerator, CropModelEnv
)


_datadir = os.path.join(_gymdir, "data")
_syncfile = os.path.join(_datadir, "Synchroniser.json")


def _read_resource(name, apsimx_dir: Optional[str] = _apsimxdir):
    if not (isinstance(apsimx_dir, str) and os.path.isdir(apsimx_dir)):
        return {}
    resource = ApsimXFile(os.path.join(apsimx_dir, "Models", "Resources",
                                       f"{name}.json"))
    return resource.find(name)


class ApsimXWeatherFile(BaseWeatherFile):
    r"""Container for ApsimX weather data."""

    _default_ext = ".met"
    _power_names = {
        "radn": "ALLSKY_SFC_SW_DWN",
        "maxt": "T2M_MAX",
        "mint": "T2M_MIN",
        "rain": "PRECTOTCORR",
        "vp": "T2MDEW",
    }
    _conv = {
        # From PCSE
        # Allen, R.G., Pereira, L.S., Raes, D. and Smith, M. (1998) Crop
        #     evapotranspiration. Guidelines for computing crop water
        #     requirements, FAO irrigation and drainage paper 56)
        "vp": lambda x: 6.108 * np.exp((17.27 * x) / (x + 237.3)),  # hPa
    }
    _units = {
        "radn": "MJ/m^2",
        "maxt": "oC",
        "mint": "oC",
        "rain": "mm",
        "vp": "hPa",
        "tav": "oC",
        "amp": "oC",
        "latitude": "decimal degrees",
        "longitude": "decimal degrees",
        "elevation": "m",
    }

    @classmethod
    def _read(cls, fname: str):
        r"""Read a model input file.

        Args:
            fname: Path to file to read.

        Returns:
            object: File contents.

        """
        out = {
            "constants": {},
            "units": {},
        }
        with open(fname, "r") as fd:
            for line in fd:
                if line.startswith("[weather.met.weather]"):
                    break
            for line in fd:
                if line.startswith("!"):
                    continue
                elif line.startswith("year"):
                    names = line.split()
                    for k, x in zip(names, fd.readline().split()):
                        out["units"][k] = x.strip("()")
                    out["columns"] = pd.read_csv(
                        fd, sep=r"\s+", names=names,
                    )
                else:
                    pattern = (
                        r"(?P<name>\w+)\s+\=\s+(?P<value>[+-]?\d+(\.\d+)?)\s+"
                        r"\((?P<units>(\w[\w\/\^ ]*)?)\)"
                    )
                    match = re.search(pattern, line)
                    if not match:
                        raise ValueError(f"Failed to parse .met line: "
                                         f"\"{line}\"")
                    match = match.groupdict()
                    if match["units"]:
                        out["units"][match["name"]] = match["units"]
                    out["constants"][match["name"]] = match["value"]
        return out

    @classmethod
    def _write(cls, fname: str, contents):
        r"""Read a model input file.

        Args:
            fname: Path to file to read.
            contents: File contents to write.

        """
        out = ["[weather.met.weather]"]
        if "constants" in contents:
            for k, v in contents["constants"].items():
                out.append(f"{k} = {v} ({contents['units'].get(k, '')})")
        column_order = ["year", "day"]
        column_order += [k for k in contents["columns"].columns
                         if k not in column_order]
        units = {
            k: f"({contents['units'].get(k, '')})"
            for k in column_order
        }
        col_space = {k: max(len(k), len(v)) for k, v in units.items()}
        for k in column_order:
            v = units[k]
            pad = (col_space[k] - len(v)) * " "
            units[k] += pad
        head, body = contents["columns"].to_string(
            index=False, col_space=col_space, columns=column_order,
        ).split("\n", maxsplit=1)
        out.append(head)
        out.append(" " + " ".join(list(units.values())))
        out.append(body)
        with open(fname, "w") as fd:
            fd.write("\n".join(out))

    @classmethod
    def _from_power(cls, src: dict):
        r"""Convert NASA power data into the correct format for this
        file.

        Args:
            src: NASA power data.

        Returns:
            Converted data.

        """
        fill_value = float(src["header"]["fill_value"])
        out = {"units": cls._units.copy()}
        out["constants"] = {
            "latitude": float(src["geometry"]["coordinates"][0]),
            "longitude": float(src["geometry"]["coordinates"][1]),
            "elevation": float(src["geometry"]["coordinates"][2]),
            "tav": np.mean(
                pd.Series(src["properties"]["parameter"]["T2M"])),
        }
        # description = [src["header"]["title"]]
        columns = {}
        for k, v in cls._power_names.items():
            s = pd.Series(src["properties"]["parameter"][v])
            s[s == fill_value] = np.nan
            columns[k] = s
        for k, v in cls._conv.items():
            columns[k] = v(columns[k])
        columns = pd.DataFrame(columns)
        date = pd.to_datetime(columns.index, format="%Y%m%d")
        columns["year"] = date.year
        columns["day"] = date.dayofyear
        ix = columns.isnull().any(axis=1)
        columns = columns[~ix]
        out["columns"] = columns
        return out

    @readonly_cached_property
    def dates(self) -> np.ndarray:
        r"""np.ndarray: Dates covered by this file."""
        return (
            (self.contents["columns"]["year"].to_numpy() - 1970).astype(
                "datetime64[Y]")
            + (self.contents["columns"]["day"].to_numpy() - 1).astype(
                "timedelta64[D]")
        )

    @readonly_cached_property
    def latitude(self) -> float:
        r"""float: Latitude (degrees)."""
        return self.contents["constants"]["latitude"]

    @readonly_cached_property
    def longitude(self) -> float:
        r"""float: Longitude (degrees)."""
        return self.contents["constants"]["longitude"]

    def _make_interactive(self, actions: list):
        r"""Modify this file to make it interactive.

        Args:
            actions: List of actions that should be enabled.

        """
        pass


class ApsimXFile(CropModelFile):
    r"""Container for manipulating .apsimx model files.

    Args:
        fname: Path to a .apsimx model file.
        generated: If True, this file was generated.
        contents: Contents to initialize the file with.

    """

    ACTION_NODES = dict({
        "sow": {
            "conflicts": [
                {
                    "contains": {"$type": "Models.Manager, Models"},
                    "calls": "Sow",
                },
            ],
        },
        "harvest": {
            "conflicts": [
                {
                    "contains": {"$type": "Models.Manager, Models"},
                    "calls": "Harvest",
                },
            ],
        },
        "irrigate": {
            "parent": {"contains": {"Name": "Field"}},
            "contains": {"Name": "Irrigation"},
            "required": True,
            # "default": os.path.join(_datadir, "Irrigate.json"),
            "default": {
                "$type": "Models.Irrigation, Models",
                "Name": "Irrigation",
                "ResourceName": "Irrigation",
                "Children": [],
                "Enabled": True,
                "ReadOnly": False,
            },
        }
    }, ** {
        k: {
            "parent": {"contains": {"Name": "Field"}},
            "contains": {"Name": "Fertiliser"},
            "required": True,
            # "default": os.path.join(_datadir, "Fertilize.json"),
            "default": {
                "$type": "Models.Fertiliser, Models",
                "Name": "Fertiliser",
                "ResourceName": "Fertiliser",
                "Children": [],
                "Enabled": True,
                "ReadOnly": False,
            },
        }
        for k in ["fertilize", "nitrogen", "calcium", "phosphorus"]
    })
    PARAM_NODES = {
        "duration": False,
        "season_length": False,
        "output_vars": {
            "contains": {
                "$type": "Models.Report, Models",
            },
            # "nested": {
            #     "EventNames": {"contains": "EndOfDay"},
            # },
            "field": "VariableNames",
            "fget": lambda x: [xx for xx in x if " as " not in xx],
        },
        "crop_name": {
            "anyOf": [
                {
                    "contains": {
                        "$type": "Models.PMF.Plant, Models"
                    },
                    "field": "Name",  # "ResourceName"?
                },
                {
                    "contains": {"Name": "SowOrHarvestByDate"},
                    "parameter": "CropName",
                },
                {
                    "calls": "Sow",
                    "parameter": "Crop",
                },
                {
                    "calls": "Harvest",
                    "parameter": "Crop",
                },
            ],
            "fget": lambda x: x.lower(),
        },
        "crop_variety": {
            "contains": {
                "$type": "Models.Manager, Models",
            },
            "calls": "Sow",
            "parameter": "CultivarName",
        },
        "year": {
            "fget": lambda x: x.year,
            "anyOf": [
                {"internal": "start_time",
                 "fset_prev": lambda x, prev: prev.replace(year=x)},
                {"internal": "end_time",
                 "fset_prev": lambda x, prev: prev.replace(year=x)},
                {"internal": "sow_date",
                 "fset_prev": lambda x, prev: prev.replace(year=x)},
                {"internal": "harvest_date",
                 "fset_prev": lambda x, prev: prev.replace(year=x)},
            ],
        },
        "start_time": {
            "contains": {"Name": "Clock"},
            "field": "Start",
            "fget": datetime.datetime.fromisoformat,
        },
        "end_time": {
            "contains": {"Name": "Clock"},
            "field": "End",
            "fget": datetime.datetime.fromisoformat,
        },
        "weather_file": {
            "contains": {
                "$type": "Models.Climate.Weather, Models",
            },
            "field": "FileName",
            "fget": lambda x: x.replace("%root%", _apsimxdir),
        },
        # "soil_file": {
        # }
        "latitude": {
            "contains": {
                "$type": "Models.Soils.Soil, Models",
            },
            "field": "Latitude"
        },
        "longitude": {
            "contains": {
                "$type": "Models.Soils.Soil, Models",
            },
            "field": "Longitude",
        },
        "field_area": {
            "contains": {
                "$type": "Models.Core.Zone, Models",
            },
            "field": "Area",
        },
        "sow_date": {
            "parent": {"contains": {"Name": "Field"}},
            "contains": {"Name": "SowOrHarvestByDate"},
            "parameter": "SowingDate",
            "default": os.path.join(_datadir, "SowOrHarvestByDate.json"),
            "fget": datetime.datetime.fromisoformat,
            "conflicts": [
                {
                    "$type": "Models.Manager, Models",
                    "calls": "Sow",
                },
            ],
            "parameter_properties": {
                "CropName": "formal_crop_name",
                "CultivarName": "crop_variety",
            },
        },
        "harvest_date": {
            "parent": {"contains": {"Name": "Field"}},
            "contains": {"Name": "SowOrHarvestByDate"},
            "parameter": "HarvestDate",
            "default": os.path.join(_datadir, "SowOrHarvestByDate.json"),
            "fget": datetime.datetime.fromisoformat,
            "conflicts": [
                {
                    "$type": "Models.Manager, Models",
                    "calls": "Harvest",
                },
            ],
            "parameter_properties": {
                "CropName": "formal_crop_name",
                "CultivarName": "crop_variety",
            },
        },
    }

    @readonly_cached_property
    def parameter_nodes(self):
        r"""dict: Previously loaded parameter nodes."""
        return {}

    @classmethod
    def crop2fname(cls, crop_name: str,
                   model_dir: Optional[str] = None) -> str:
        r"""Locate an input model file for a given crop name.

        Args:
            crop_name: Crop name.
            model_dir: Directory containing the model.

        Returns:
            str: Model input file for the specified crop.

        """
        if model_dir is None:
            model_dir = _apsimxdir
        examples_dir = os.path.join(model_dir, "Examples")
        for x in [crop_name, crop_name.title()]:
            fname = os.path.join(examples_dir, f"{x}.apsimx")
            if os.path.isfile(fname):
                return fname
        raise ValueError(f"Could not locate a model file for crop "
                         f"\"{crop_name}\".")

    @property
    def formal_crop_name(self) -> str:
        r"""str: Crop name used for resources."""
        # TODO: Lookup resource?
        return self.crop_name.title()

    def _get_external_name(self, name: str) -> str:
        r"""Get the external variable name from the internal variable
        name.

        Args:
            name: Internal parameter name.

        Returns:
            str: Parameter name.

        """
        # TODO: Map variables in PARAM_NODES?
        if name.startswith(f"[{self.formal_crop_name}]"):
            return name.replace(f"[{self.formal_crop_name}]", "[CROP]")
        return name

    def _get_internal_name(self, name: str) -> str:
        r"""Get the internal variable name from a model parameter name.

        Args:
            name: Parameter name.

        Returns:
            str: Internal parameter name.

        """
        if name not in self.PARAM_NODES:
            if name.startswith("[CROP]"):
                return name.replace("[CROP]",
                                    f"[{self.formal_crop_name}]")
            return name
        info = self.PARAM_NODES[name]
        node = self.find_parameter(name, required=True)
        if "parameter" in info:
            names = [x["Key"] for x in node["Parameters"]]
            field = info["parameter"]
            if field not in names:
                raise KeyError(
                    f"Parameter \"{field}\" not present in node "
                    f"\"{node['Name']}\" for parameter \"{name}\" "
                    f"(Parameters = {names})"
                )
            idx = names.index(field)
            out = f"[{node['Name']}].Parameters[{idx}].Value"
        else:
            out = f"[{node['Name']}].{info['field']}"
        return out

    @classmethod
    def _get_node_parameter(cls, node: dict, name: str) -> Any:
        r"""Extract a node parameter from a parameter list.

        Args:
            node: Node containing parameters
            name: Parameter name.

        Returns:
            Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        for x in node["Parameters"]:
            if isinstance(name, set) and x["Key"] in name:
                return x["Value"]
            elif x["Key"] == name:
                return x["Value"]
        raise KeyError(f"No parameter named \"{name}\" in {node}")

    @classmethod
    def _set_node_parameter(cls, node: dict, name: str, value: Any):
        r"""Set a node parameter.

        Args:
            node: Node containing parameters
            name: Parameter name.
            value: Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        for x in node["Parameters"]:
            if isinstance(name, set) and x["Key"] in name:
                x["Value"] = value
                return
            elif x["Key"] == name:
                x["Value"] = value
                return
        raise KeyError(f"No parameter named \"{name}\" in {node}")

    @classmethod
    def _get_parameter(cls, node: dict, info: dict):
        out = None
        if "parameter" in info:
            out = cls._get_node_parameter(node, info["parameter"])
        elif "field" in info:
            out = node[info["field"]]
        elif "internal" in info:
            out = cls._get_parameter(node, cls.PARAM_NODES[info["internal"]])
        elif "anyOf" in info:
            for x in info["anyOf"]:
                if cls.node_matches(node, **x):
                    out = cls._get_parameter(node, x)
                    break
            else:
                errors = []
                for x in info["anyOf"]:
                    cls.node_matches(node, errors=errors, **x)
                raise ValueError(
                    "Node does not match the requirements:\n  "
                    + "\n  ".join(errors)
                )
        else:
            raise NotImplementedError(f"Invalid info {info}")
        if "fget" in info:
            try:
                out = info["fget"](out)
            except ValueError as e:
                raise KeyError(e)
        return out

    @classmethod
    def _set_parameter(cls, node: dict, info: dict, value: Any):
        if isinstance(value, (datetime.datetime, datetime.date)):
            value = value.isoformat()
        if "fset" in info:
            try:
                value = info["fset"](value)
            except ValueError as e:
                raise KeyError(e)
        if "fset_prev" in info:
            prev = cls._get_parameter(node, info)
            value = info["fset_prev"](value, prev)
        if "parameter" in info:
            cls._set_node_parameter(node, info["parameter"], value)
        elif "field" in info:
            node[info["field"]] = value
        elif "internal" in info:
            cls._set_parameter(node, cls.PARAM_NODES[info["internal"]],
                               value)
        elif "anyOf" in info:
            for x in info["anyOf"]:
                if cls.node_matches(node, **x):
                    cls._set_parameter(node, x, value)
                    break
            else:
                errors = []
                for x in info["anyOf"]:
                    cls.node_matches(node, errors=errors, **x)
                raise ValueError(
                    "Node does not match the requirements:\n  "
                    + "\n  ".join(errors)
                )
        else:
            raise NotImplementedError(f"Invalid info {info}")

    def _get(self, name: str):
        r"""Get a parameter from the model file.

        Args:
            name: Parameter name.

        Returns:
            Parameter value.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        node = self.find_parameter(name, required=True)
        info = self.PARAM_NODES[name]
        try:
            return self._get_parameter(node, info)
        except KeyError as e:
            raise KeyError(f"{name}: {e}")

    def _set(self, name: str, value: Any,
             info: Optional[dict] = None) -> Any:
        r"""Set a parameter in the model file.

        Args:
            name: Parameter name.
            value: Parameter value.
            info: Information about how to locate the parameter.

        Raises:
            KeyError: If name is not a valid parameter name.

        """
        add_missing = (info is None)
        if info is None:
            info = self.PARAM_NODES[name]
        if info is False:
            return
        try:
            anyset = False
            for xnode in self.findall_parameters(name, info=info):
                try:
                    self._set_parameter(xnode, info, value)
                    anyset = True
                except KeyError:
                    continue
            if not anyset:
                node = self.find_parameter(
                    name, required=True,
                    add_missing=add_missing,
                    info=info,
                )
                self._set_parameter(node, info, value)
        except KeyError as e:
            raise KeyError(f"{name}: {e}")

    @classmethod
    def _read(cls, fname: str):
        r"""Read a model input file.

        Args:
            fname: Path to file to read.

        Returns:
            object: File contents.

        """
        with open(fname, 'r') as fd:
            out = json.load(fd)
        return out

    @classmethod
    def _write(cls, fname: str, contents):
        r"""Read a model input file.

        Args:
            fname: Path to file to read.
            contents: File contents to write.

        """
        with open(fname, "w") as fd:
            json.dump(contents, fd, indent="    ")

    @readonly_cached_property
    def is_interactive(self):
        r"""bool: True if the .apsimx model is interactive."""
        return bool(self.find("Synchroniser"))

    def disable_parameter_conflicts(self, name: str,
                                    info: Optional[dict] = None,
                                    node: Optional[dict] = None):
        r"""Disable nodes that conflict with a parameter/action.

        Args:
            name: Action name.
            info: Information about the parameter.
            node: Parameter/action node to avoid disabling.

        """
        if info is None:
            info = (
                self.ACTION_NODES[name] if name in self.ACTION_NODES
                else self.PARAM_NODES[name]
            )
        if node is None and self.includes_constraints(info):
            node = self.find_parameter(name, info=info)
        for vconflict in info.get("conflicts", []):
            for x in self.findall(requirements=vconflict):
                if x == node:
                    continue
                logger.info(
                    f"Disabling node \"{x['Name']}\" which "
                    f"conflicts with parameter/action \"{name}\""
                )
                x["Enabled"] = False
                self.generated = False

    def add_parameter(self, name: str, info: Optional[dict] = None,
                      parent: Optional[dict] = None) -> dict:
        r"""Add a node to facilitate use of a parameter/action if it
        is missing.

        Args:
            name: Action name.
            info: Information about how to add the parameter.
            parent: Parent node that the action node should be added to
                if it is missing.

        Returns:
            dict: Action node.

        """
        if info is None:
            info = (
                self.ACTION_NODES[name] if name in self.ACTION_NODES
                else self.PARAM_NODES[name]
            )
        # Do conflicts first before adding the default so that the
        # default is not disabled by mistake
        self.disable_parameter_conflicts(name, info=info, node={})
        default = info.get("default", None)
        if isinstance(default, str):
            default = ApsimXFile(default).contents
        if not default:
            logger.warning(
                f"No default node registered for parameter/action "
                f"\"{name}\""
            )
        if parent is None and "parent" in info:
            parent = self.find(
                parent=True, requirements=info["parent"]
            )
        if not parent:
            logger.warning(
                f"No parent node registered for parameter/action "
                f"\"{name}\""
            )
        node = None
        if parent and default:
            parent["Children"].append(default.copy())
            node = parent["Children"][-1]
            node["Enabled"] = True
            self.generated = False
        if node:
            for k, v in info.get("parameter_properties", {}).items():
                value = getattr(self, v, None)
                if value is not None:
                    self._set_node_parameter(node, k, value)
                    self.generated = False
        return node

    def disable_action(self, name: str, **kwargs):
        r"""Disable any nodes that automatically control an action.

        Args:
            name: Action name.
            **kwargs: Additional keyword arguments are passed to
                find_parameter.

        """
        node = self.find_parameter(name, **kwargs)
        if node is not None:
            node["Enabled"] = False
            self.generated = False

    def enable_action(self, name: str, parent: Optional[dict] = None,
                      **kwargs):
        r"""Enable any nodes that automatically control an action.

        Args:
            name: Action name.
            parent: Parent node that the action node should be added to
                if it is missing.
            **kwargs: Additional keyword arguments are passed to
                find_parameter.

        Returns:
            dict: Action node.

        """
        info = self.ACTION_NODES[name]
        if not self.includes_constraints(info):
            self.disable_parameter_conflicts(name, info=info, node={})
            return
        return self.find_parameter(
            name, info=info,
            add_missing=(parent if parent else True),
            current=parent, **kwargs
        )

    def disable(self, name: str, **kwargs):
        r"""Disable a node in the file if it exists.

        Args:
            name: Name of the node to disable.
            **kwargs: Additional keyword arguments are passed to find.

        """
        node = self.find(name, **kwargs)
        if node is not None:
            node["Enabled"] = False
            self.generated = False

    @classmethod
    def includes_constraints(cls, info: dict) -> bool:
        r"""Check if a set of node requirements constrain the node.

        Args:
            info: Node requirements.

        Returns:
            bool: True if info constrains the node, False otherwise.

        """
        return any(k in info for k in [
            "name", "field", "parameter", "internal", "contains",
            "equals", "fvalid", "calls", "anyOf", "nested"])

    @classmethod
    def node_matches(cls, node: Any,
                     errors: Optional[list] = None,
                     name: Optional[str] = None,
                     field: Optional[str] = None,
                     parameter: Optional[str] = None,
                     internal: Optional[str] = None,
                     contains: Optional[Union[list, set, dict]] = None,
                     equals: Optional[Any] = None,
                     fvalid: Optional[Callable] = None,
                     calls: Optional[str] = None,
                     anyOf: Optional[list] = None,
                     nested: Optional[dict] = None,
                     **kwargs) -> bool:
        r"""Check if a node matches the specified requirements.

        Args:
            node: Node to check.
            errors: If a list is provided, errors will be added to this
                list.
            name: Name that the node must have.
            field: Name of a field that must be present.
            parameter: Name of a parameter that must be present.
            contains: Fields/elements that the node must contain. If a
                set is provided, only one of the elements must be
                present. If a dict is provided, the values in the node
                must match the values in the provided dict.
            equals: Value that the node must be equivalent to.
            fvalid: Function that returns True if the node is valid, and
                False otherwise.
            calls: Name of a function called in the node code block.
            anyOf: List of kwargs for node_matches that should be
                checked. If the node satisfies any of these requirements,
                True will be returned.
            nested: Set of requirements for individual fields.
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            bool: True if the node matches, False otherwise.

        """

        def add_error(msg):
            if not isinstance(errors, list):
                return False
            errors.append(msg)
            return True

        if anyOf:
            if isinstance(errors, list):
                xerrors = []
                if not any(cls.node_matches(node, errors=xerrors, **x)
                           for x in anyOf):
                    errors += xerrors
            else:
                if not any(cls.node_matches(node, **x) for x in anyOf):
                    return False
        if name is not None and node.get("Name", None) != name:
            if not add_error(f"{node} name is not {name}"):
                return False
        if field is not None and field not in node:
            if not add_error(f"{node} is missing field \"{field}\""):
                return False
        if ((parameter is not None
             and not any(x["Key"] == parameter
                         for x in node.get("Parameters", [])))):
            if not add_error(f"{node} is missing parameter \"{parameter}\""):
                return False
        if internal:
            if not cls.node_matches(node, **cls.PARAM_NODES[internal]):
                return False
        if contains:
            if isinstance(contains, str):
                contains = [contains]
            if isinstance(contains, set):
                missing = (
                    list(contains)
                    if not any(k in node for k in contains)
                    else []
                )
            elif isinstance(contains, dict):
                missing = [k for k in contains.keys() if k not in node]
            else:
                assert isinstance(contains, list)
                missing = [k for k in contains if k not in node]
            if missing and not add_error(f"Missing {missing}"):
                return False
            if isinstance(contains, dict):
                for k, v in contains.items():
                    if k not in node:
                        continue
                    if node[k] != v and not add_error(
                            f"{k}: {node[k]} != {v}"):
                        return False
        if equals and node != equals:
            if not add_error(f"{node} != {equals}"):
                return False
        if fvalid and not fvalid(node):
            if not add_error(f"{node} fails function {fvalid}"):
                return False
        if ((calls and not (
                isinstance(node, dict)
                and any(f"{calls}(" in x
                        for x in node.get("CodeArray", []))))):
            if not add_error(f"{node} does not call \"{calls}\""):
                return False
        if nested:
            missing = [k for k in nested.keys() if k not in node]
            if missing and not add_error(f"Missing {missing}"):
                return False
            for k, v in nested.items():
                if k not in node:
                    continue
                if isinstance(errors, list):
                    verrors = []
                    cls.node_matches(node[k], errors=verrors, **v)
                    errors += [f"{k}: {x}" for x in verrors]
                elif not cls.node_matches(node[k], **v):
                    return False
        # if kwargs:
        #     missing = [k for k in kwargs.keys() if k not in node]
        #     if missing and not add_error(f"Missing {missing}"):
        #         return False
        #     for k, v in kwargs.items():
        #         if k not in node:
        #             continue
        #         if node[k] != v and not add_error(f"{k}: {node[k]} != {v}"):
        #             return False
        if errors:
            return False
        return True

    def findall_parameters(self, name: str, info: Optional[dict] = None,
                           **kwargs):
        r"""Find all parameters nodes in this file matching the
        parameter info.

        Args:
            name: Parameter name.
            info: Information about how to locate the parameter.
            **kwargs: Additional keyword arguments are passed to findall.

        Yields:
            dict: The nodes matching the parameter info.

        Raises:
            KeyError: If info not provided and name is not a valid
                parameter/action.

        """
        if info is None:
            if ((name not in self.PARAM_NODES
                 and name not in self.ACTION_NODES)):
                raise KeyError(f"No node registered for parameter "
                               f"\"{name}\"")
            info = (
                self.PARAM_NODES[name] if name in self.PARAM_NODES
                else self.ACTION_NODES[name]
            )
        for node in self.findall(requirements=info, **kwargs):
            yield node

    def find_parameter(self, name: str,
                       add_missing: Optional[Union[bool, dict]] = False,
                       info: Optional[dict] = None,
                       **kwargs) -> dict:
        r"""Find a parameter node in the file.

        Args:
            name: Parameter name.
            add_missing: If True or dict, the default for the parameter
                will be added if it cannot be located. If a dict is
                provided, the parameter default will be added to this if
                the parameter cannot be located.
            info: Information about how to locate the parameter.
            **kwargs: Additional keyword arguments are passed to find.

        Returns:
            dict: The node matching the specified name. Empty if no
                node can be found.

        Raises:
            KeyError: If required is True and the node cannot be located.

        """
        if info is None:
            if ((name not in self.PARAM_NODES
                 and name not in self.ACTION_NODES)):
                if not kwargs.get("required", False):
                    return {}
                raise KeyError(f"No node registered for parameter "
                               f"\"{name}\"")
            info = (
                self.PARAM_NODES[name] if name in self.PARAM_NODES
                else self.ACTION_NODES[name]
            )
        if info is False:
            raise KeyError(f"Ignored parameter \"{name}\"")
        if ((name in self.parameter_nodes
             and not kwargs.get("parent", False))):
            return self.parameter_nodes[name]
        node = None
        try:
            node = self.find(requirements=info, **kwargs)
            if add_missing and not node:
                assert not kwargs.get("parent", False)  # corner case
                node = self.add_parameter(
                    name, info=info, parent=(
                        add_missing if isinstance(add_missing, dict)
                        else None
                    ),
                )
            if node and not kwargs.get("parent", False):
                self.parameter_nodes[name] = node
            return node
        except KeyError as e:
            if add_missing and not node:
                assert not kwargs.get("parent", False)  # corner case
                node = self.add_parameter(name)
                if node:
                    return node
            raise KeyError(f"{name}: {e}")

    def findall(self, name: Optional[str] = None,
                current: Optional[dict] = None,
                parent: Optional[Union[bool, dict]] = False,
                requirements: Optional[dict] = None) -> list:
        r"""Find a node in the file.

        Args:
            name: Name of the node to find.
            current: The current node being searched.
            parent: The parent node. If True, the parent node will be
                returned.
            requirements: Set of requirements that the node must
                satisfy (see node_matches for a description of the
                available options).

        Yields:
            dict: All nodes matching the specified name.

        """
        requirements = requirements or {}
        assert name is not None or requirements
        if current is None:
            current = self.contents
        if self.node_matches(current, name=name, **requirements):
            yield parent if parent else current
        for x in current.get("Children", []):
            for out in self.findall(
                    name=name, current=x,
                    parent=(current if parent else False),
                    requirements=requirements
            ):
                yield out

    def find(self, name: Optional[str] = None,
             current: Optional[dict] = None,
             parent: Optional[Union[bool, dict]] = False,
             required: Optional[bool] = False,
             requirements: Optional[dict] = None) -> dict:
        r"""Find a node in the file.

        Args:
            name: Name of the node to find.
            current: The current node being searched.
            parent: The parent node. If True, the parent node will be
                returned.
            required: If True, an error will be raised if the node
                cannot be located.
            requirements: Set of requirements that the node must
                satisfy (see node_matches for a description of the
                available options).

        Returns:
            dict: The node matching the specified name. Empty if no
                node can be found.

        Raises:
            KeyError: If required is True and the node cannot be located.

        """
        requirements = requirements or {}
        assert name is not None or requirements
        if current is None:
            current = self.contents
        if self.node_matches(current, name=name, **requirements):
            out = parent if parent else current
            return out
        for x in current.get("Children", []):
            out = self.find(name=name, current=x,
                            parent=(current if parent else False),
                            requirements=requirements)
            if out:
                return out
        if required:
            msg = ""
            if name is not None:
                msg += f" with \"Name\" {name}"
            if requirements:
                msg += f" matching requirements {requirements}"
            raise KeyError(f"Could not locate a node{msg}")
        return {}

    def _make_interactive(self, actions: list):
        r"""Modify this file to make it interactive.

        Args:
            actions: List of actions that should be enabled.

        """
        sync = ApsimXFile(_syncfile)
        field = self.find("Field", required=True)
        for k, v in self.ACTION_NODES.items():
            parent = None
            if "parent" in v and self.node_matches(field, **v["parent"]):
                parent = field
            if k not in actions:
                if v.get("required", False):
                    self.find_parameter(
                        k, current=parent, required=True,
                        add_missing=(parent if parent else True),
                    )
                else:
                    self.disable_action(k)
                continue
            self.enable_action(k, parent=parent)
        # Do sync last to avoid it being disabled
        field["Children"].append(copy.deepcopy(sync.contents))


_fertilizer_node = _read_resource("Fertiliser")


def _fertilizer_action(name: Optional[str] = None,
                       solutes: Optional[List[str]] = None,
                       **kwargs):
    if name is None:
        fullname = "fertilizer"
    else:
        fullname = f"{name} fertilizer"
    if solutes:
        types = []
        for x in _fertilizer_node.get("Children", []):
            for i in range(1, 10):
                field = f"Solute{i}Name"
                if field not in x:
                    break
                if x[field] in solutes:
                    types.append(x["Name"])
    else:
        types = [x["Name"] for x in _fertilizer_node.get("Children", [])]
    out = {
        "description": (
            "Apply {amount} {amount_units} "
            + fullname + " in the form of {type}"
        ),
        "action_param": "amount",
        "param_desc": {
            "amount": {
                "type": "number",
                "units": "kg/ha",
                "min": 0.0,
                "max": 8.0,
            },
            "type": {
                "type": "string",
                "default": types[0] if types else name,
                "enum": types,
            },
            # "depth": {
            #     "type": "number",
            #     "units": "mm",
            # },
            # "depthBottom": {
            #     "type": "number",
            #     "units": "mm",
            # },
        },
    }
    out.update(**kwargs)
    return out


# connect -> ok
# paused -> resume/get/set
# finished -> ok
class ApsimXEngine(CropModelEngine):
    r"""Class for managing communication with an APSIMX server running
    in another process.

    Args:
        model_file: Path to a .apsimx model input file.
        model_dir: Path to the directory containing APSIMX installation.
        **kwargs: Additional keyword arguments are passed to the
            CropModelEngine constructor.

    """

    STATUS_MESSAGES = [
        "connect", "finished", "error", "recoverable_error",
    ]
    ERROR_MESSAGES = [
        "error", "recoverable_error",
    ]
    INPUT_FILE_TYPE = ApsimXFile
    WEATHER_FILE_TYPE = ApsimXWeatherFile
    AVAILABLE_ACTION_MAP = {
        "sow": {
            "description": (
                "Sow a {crop_variety} {crop_name} crop at a density "
                "of {population} {population_units} with a sowing "
                "depth of {sowingDepth} {sowingDepth_units} and a row "
                "spacing of {rowSpacing} {rowSpacing_units}"
            ),
            "action_param": None,  # Boolean
            "param_desc": {
                "crop_name": {
                    "type": "string",
                },
                "crop_variety": {
                    "type": "string",
                    "default": "Hartog",
                },
                "population": {
                    "type": "number",
                    "units": "seeds per square meter",  # seeds/m²
                    "default": 5.0,
                },
                "sowingDepth": {
                    "type": "number",
                    "units": "mm",
                    "default": 50.0,
                },
                "rowSpacing": {
                    "type": "number",
                    "units": "mm",
                    "default": 1000.0,
                },
            },
            "num_levels": -1,  # Boolean
        },
        "harvest": {
            "description": "Harvest the {crop_name} crop",
            "action_param": None,  # Boolean
            "param_desc": {
                "crop_name": {
                    "type": "string",
                    "default": "",
                },
            },
            "num_levels": -1,  # Boolean
        },
        "tillage": {
            "description": "Till the field using {type} tillage",
            "action_param": "type",
            "param_desc": {
                "type": {
                    "type": "string",
                    "enum": ["chisel", "disc", "planter", "burn"],
                },
            },
        },
        "fertilize": _fertilizer_action(),
        "nitrogen": _fertilizer_action(
            "nitrogen", ["NO3", "NH4", "Urea"],
            alias="N",  # cost=0.46,  # $/kg
        ),
        "calcium": _fertilizer_action(
            "calcium", ["Ca"],
            alias="Ca",
        ),
        "phosphorus": _fertilizer_action(
            "phosphorus", ["RockP", "LabileP", "BandedP"],
            alias="P",
        ),
        "irrigate": {
            "alias": "water",
            "description": (
                "Irrigate with {amount} {amount_units} of water."
            ),
            "action_param": "amount",
            # 1 mm over 1 ha == 10000 L, $20 per 10000 L
            # "cost": 20.0,  # $/mm/ha
            "param_desc": {
                "amount": {
                    "type": "number",
                    "units": "mm",
                    "min": 0.0,
                    "max": 20.0,
                },
            },
        },
    }

    def __init__(
            self,
            model_file: Optional[str] = None,
            model_dir: Optional[str] = None,
            **kwargs
    ):
        if model_dir is None:
            model_dir = _apsimxdir
        self.apsim_srv = os.path.join(
            model_dir, "bin", "Debug", "net8.0", "ApsimZMQServer.dll")
        self.context = None
        self.socket = None
        self.port = None
        self.process = None
        self.stdout_pipe = None
        self.stderr_pipe = None
        self._status = None
        self._current_time = None
        if not (isinstance(model_dir, str) and os.path.isdir(model_dir)):
            raise RuntimeError(f"APSIMX directory does not "
                               f"exist: \"{model_dir}\"")
        if not os.path.isfile(self.apsim_srv):
            raise RuntimeError(f"APSIMX server executable does not "
                               f"exist: \"{self.apsim_srv}\"")
        if model_file and not (os.path.isfile(model_file)
                               or os.path.dirname(model_file)):
            model_file = os.path.join("Examples", model_file)
        super().__init__(model_file=model_file, model_dir=model_dir,
                         **kwargs)
        if not self.output_dir:
            # ApsimX saves output to the directory containing the
            # model input file
            self.output_dir = os.path.dirname(self.model.fname)
        self.products += [
            self.output_file,
            f"{self.output_file}-shm",
            f"{self.output_file}-wal"
        ]

    @property
    def is_running(self):
        r"""bool: True if the model engine is still running."""
        return (self.process is not None
                and self.process.poll() is None)

    @property
    def is_operable(self):
        r"""bool: True if the model engine is running and functioning."""
        if not super().is_operable:
            return False
        return (self._status not in ["finished", "error", "terminated"])

    @property
    def current_time(self):
        r"""datetime.datetime: Current simulation time."""
        if self._current_time is None:
            if not self.is_operable:
                return self.start_time
            return self.get("[Clock].Today")
        return self._current_time

    @property
    def status(self):
        r"""str: Current simulation status."""
        if self._status is None and self.socket is not None:
            self._status = self.socket.recv_string()
            if self._status == "paused":
                self._current_time = None
                self.current_time
                logger.debug(f"Simulation waiting at {self.current_time}")
            elif self._status in self.STATUS_MESSAGES:
                out = self._status
                self.send_command("ok")
                return out
            else:
                raise NotImplementedError(f"Unsupported status message: "
                                          f"\"{self._status}\"")
        return self._status

    @property
    def output_file(self):
        r"""str: Path to the .db output file that will be produced."""
        return os.path.join(
            os.path.splitext(self.model.fname)[0] + ".db")

    def get_output_vars(self) -> List[str]:
        r"""Get the output variables specified by the model file.

        Returns:
            list: Output variables

        """
        out = super().get_output_vars()
        return [self.model._get_external_name(x) for x in out]

    def _start(self):
        r"""Start a listening server on a random port."""
        self._current_time = None
        self._status = None
        self.context = zmq.Context()
        if self.socket is None:
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind("tcp://0.0.0.0:0")
            self.port = self.socket.getsockopt(
                zmq.LAST_ENDPOINT).decode().split(":")[-1]
        logger.info(f"Running model \"{self.model.fname}\"")
        logger.info(
            f"Listening on: {self.socket.getsockopt(zmq.LAST_ENDPOINT)}")
        self.process = subprocess.Popen([
            "dotnet", self.apsim_srv,
            "-p", self.port,
            "-P", "interactive",
            "-f", self.model.fname,
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.stdout_pipe = LogPipe(
            self.process.stdout, prefix="APSIMX: ")
        self.stderr_pipe = LogPipe(
            self.process.stderr, prefix="APSIMX", level="ERROR")
        logger.info(f"Started APSIMX process id: {self.process.pid}")
        assert self.status == "connect"
        if self.status != "paused":
            self.stop(cleanup=True)
            raise AssertionError(f"Server is not awaiting instructions: "
                                 f"status = \"{self.status}\"")
        if self.start_time is None:
            self.start_time = self.get("[Clock].Start")
        else:
            assert self.get("[Clock].Start") == self.start_time
        if self.end_time is None:
            self.end_time = self.get("[Clock].End")
        else:
            assert self.get("[Clock].End") == self.end_time

    def _stop(self):
        r"""Stop the listening server and close the communication port."""
        if self.is_operable:
            try:
                self.act("terminate")
                if self.status != "finished":
                    raise ValueError(
                        f"Status after terminate is \"{self.status}\"")
                self._status = "terminated"
            except ModelEngineError:
                pass
        if self.socket is not None:
            self.socket.close()
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=1)
                assert self.process.poll() is not None
            # process = psutil.Process(self.process.pid)
            # for proc in process.children(recursive=True):
            #     proc.kill()
            # process.kill()
        if self.stderr_pipe is not None:
            self.stderr_pipe.close()
        if self.stderr_pipe is not None:
            self.stderr_pipe.close()
        if self.context is not None:
            self.context.destroy()
        if self._status != "error":
            self.context = None
            self.socket = None
            self.port = None
            self.process = None
            self.stdout_pipe = None
            self.stderr_pipe = None

    def send_command(self, command: str, args: Optional[list] = None):
        r"""Send a command to the server process, e.g. resume/set/get.

        Args:
            command: Command to send.
            args: Additional arguments to send with the commaned.

        """
        assert self.status is not None
        self.socket.send_string(command, zmq.SNDMORE if args else 0)
        if args:
            for i, arg in enumerate(args):
                self.socket.send(msgpack.packb(arg),
                                 zmq.SNDMORE if i < len(args) - 1 else 0)
        if self.status != 'finished':
            self._status = None

    def recv_reply(self, unpack: Optional[bool] = False):
        r"""Receive a reply from the server process.

        Args:
            unpack: If True, the message will be unpacked using msgpack.

        Returns:
            object: Received message.

        """
        if unpack:
            outb = self.socket.recv()
            try:
                out = msgpack.unpackb(outb)
            except msgpack.exceptions.ExtraData:
                out = outb.decode()
                if out not in self.STATUS_MESSAGES:
                    raise
        else:
            out = self.socket.recv_string()
        if out in self.STATUS_MESSAGES:
            self._status = out
            self.send_command("ok")
            if out == "recoverable_error":
                self._status = self.recv_reply()
                assert self._status == "paused"
        return out

    def check_paused(self):
        r"""Check that the simulation server is paused."""
        if self.status != "paused":
            raise ModelEngineError(
                f"Simulation is not paused (status = {self.status})"
            )

    @contextlib.contextmanager
    def stop_on_error(self, record: Optional[tuple] = None,
                      allow_error: Optional[bool] = False):
        r"""Context manager that stops the simulation on an error.

        Args:
            record: Action to log when successful.
            allow_error: If True, a RecoverableError error will not
                result in the simulation being stopped.

        """
        self.check_paused()
        with super().stop_on_error(record=record, allow_error=allow_error):
            status = self.status
            try:
                yield
            except RecoverableError:
                raise
            except BaseException:
                self._status = "error"
                raise
            self._status = status

    def _reply_error(self, reply):
        if reply == "recoverable_error":
            error_cls = RecoverableModelEngineError
        else:
            error_cls = ModelEngineError
        return error_cls

    def _get(self, name: str):
        r"""Send a request to get the current value of a simulation state
        variable.

        Args:
            name: Name of variable to get the value of.

        Returns:
            object: Current variable value.

        """
        reply = None
        try:
            name = self.model._get_internal_name(name)
        except KeyError as e:
            raise InvalidActionError(e)
        self.send_command("get", [name])
        reply = self.recv_reply(unpack=True)
        if reply in self.ERROR_MESSAGES:
            raise self._reply_error(reply)(
                f"get for \"{name}\" received error reply "
                f"\"{reply}\""
            )
        if isinstance(reply, msgpack.ext.Timestamp):
            reply = reply.to_datetime().replace(tzinfo=None)
        return reply

    def _set(self, name: str, value):
        r"""Send a request to set a simulation state variable.

        Args:
            name: Name of the variable to update.
            value: New value for the named variable.

        """
        try:
            name = self.model._get_internal_name(name)
        except KeyError as e:
            raise InvalidActionError(e)
        if isinstance(value, (datetime.datetime, datetime.date)):
            value = value.isoformat()
        self.send_command("set", [name, value])
        reply = self.recv_reply()
        if reply != "ok":
            raise self._reply_error(reply)(
                f"set for \"{name}\" received non-ok reply "
                f"\"{reply}\""
            )

    def _act(self, action: str, param: dict):
        r"""Perform an action.

        Args:
            name: Name of the action to perform.
            param: Action parameters.

        """
        args_flat = []
        for k, v in param.items():
            args_flat += [k, v]
        self.send_command("act", [action] + args_flat)
        reply = self.recv_reply()
        if reply != "ok":
            raise self._reply_error(reply)(
               f"act for \"{action}\" received non-ok reply "
               f"\"{reply}\""
            )

    def resume(self, wait: Optional[bool] = False):
        r"""Resume the simulation.

        Args:
            wait: If True, wait for the simulation to pause.

        """
        self.check_paused()
        self.send_command("resume")
        if wait:
            self.status


class ApsimXLLMPromptGenerator(CropModelLLMPromptGenerator):
    r"""ApsimX LLM prompt generator."""

    # TODO: Verify units
    DEFAULT_DESC_MAP = {
        "[Clock].Today": (
            "Timeline", "Date/time"),
        "[CROP].LAI": (
            "Crop status", "Leaf area index"),
        "[CROP].Phenology.Zadok.Stage": (
            "Crop status", "Zadok developmental stage"),
        "[CROP].Phenology.CurrentStageName": (
            "Crop status", "Developmental stage name"),
        "[CROP].Total.Wt": (
            "Crop status", "Total crop mass (kg/ha)"),
        "[CROP].Grain.Total.Wt": (
            "Crop status", "Crop yield (g/m²)"),
        "[CROP].Grain.Protein": (
            "Crop status", "Crop grain protein content"),
        "[CROP].Grain.Size": (
            "Crop status", "Crop grain size"),
        "[CROP].Grain.Number": (
            "Crop status", "Crop grain number"),
        "[CROP].Grain.Total.N": (
            "Crop status", "Crop grain nitrogen content"),
        # "[CROP].Biomass.StorageWt": (
        #     "Crop status", "Storage organ dry matter (g/m²)"),
        "[CROP].AboveGround.Wt": (
            "Crop status", "Total above-ground crop mass"),
        "[CROP].AboveGround.N": (
            "Crop status", "Total above-ground crop nitrogen content"),
        "[Nutrient].TotalN.kgha": (
            "Soil nutrients", "Available soil nitrogen (kg/ha)"),
        "[Soil].Water.PAW": (
            "Soil & water",
            "Plant-available root-zone soil moisture (fraction)"),
        "[Fertiliser].NitrogenApplied": (
            "Cumulative actions",
            "Cumulative nitrogen applied so far (kg/ha)"),
        "[Irrigation].IrrigationApplied": (
            "Cumulative actions",
            "Cumulative irrigation depth applied so far (mm)"),
        "[Weather].Radn": (
            "Weather", "Daily solar radiation (MJ/m²/day)"),
        "[Weather].Tav": (
            "Weather", "Mean air temperature (°C)"),
        "[Weather].Rain": (
            "Weather", "Daily rainfall (mm)"),
        "[CROP].DaysAfterSowing": ("Timeline", "Days since sowing"),
    }


class ApsimXEnv(CropModelEnv):
    r"""ApsimX environment."""

    MODEL_ENGINE_CLASS = ApsimXEngine
    LLM_PROMPT_GENERATOR_CLASS = ApsimXLLMPromptGenerator
    DEFAULT_ACTIONS = ["nitrogen", "irrigate"]
    DEFAULT_REVENUE_VAR = {
        "name": "[CROP].Grain.Total.Wt",
        # "name": "Yield",  # Only includes harvested weight
        # "cost": ??,  # $/kg/ha
    }
