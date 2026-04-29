# python RL-Gym/run_interactive.py run Examples/Wheat.apsimx
import os
import json
import copy
import zmq
import msgpack
import subprocess
import contextlib
import datetime
from typing import Optional, Union
from functools import cached_property
from . import logger
from .utils import _gymdir, _apsimxdir, LogPipe
from .base import (
    RecoverableError, ModelEngineError,
    RecoverableModelEngineError,
    BaseModelFile, BaseModelEngine, BaseModelEnv,
)


_datadir = os.path.join(_gymdir, "data")
_syncfile = os.path.join(_datadir, "Synchroniser.apsimx")


class ApsimXFile(BaseModelFile):
    r"""Container for manipulating .apsimx model files.

    Args:
        fname: Path to a .apsimx model file.
        generated: If True, this file was generated.
        contents: Contents to initialize the file with.

    """

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

    @cached_property
    def is_interactive(self):
        r"""bool: True if the .apsimx model is interactive."""
        return bool(self.find("Synchroniser"))

    def disable(self, name: str):
        r"""Disable a node in the file if it exists.

        Args:
            name: Name of the node to disable.

        """
        node = self.find(name)
        if node is not None:
            node["Enabled"] = False
            self.generated = False

    def find(self, name: str, current: Optional[dict] = None,
             parent: Optional[Union[bool, dict]] = False,
             required: Optional[bool] = False) -> dict:
        r"""Find a node in the file.

        Args:
            name: Name of the node to find.
            current: The current node being searched.
            parent: The parent node. If True, the parent node will be
                returned.
            required: If True, an error will be raised if the node
                cannot be located.

        Returns:
            dict: The node matching the specified name. Empty if no
                node can be found.

        Raises:
            ValueError: If required is True and the node cannot be located.

        """
        if current is None:
            current = self.contents
        if current["Name"] == name:
            if parent:
                return parent
            return current
        for x in current.get("Children", []):
            out = self.find(name, current=x,
                            parent=(current if parent else False))
            if out:
                return out
        if required:
            raise ValueError(f"Could not locate a node with the name "
                             f"\"{name}\"")
        return {}

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
        clock = self.find("Clock", required=True)
        if start_time is not None:
            clock["Start"] = start_time.isoformat()
        if end_time is not None:
            clock["End"] = end_time.isoformat()

    def _make_interactive(self):
        r"""Modify this file to make it interactive."""
        sync = ApsimXFile(_syncfile)
        field = self.find("Field", required=True)
        field["Children"].append(copy.deepcopy(sync.contents))
        for k in ["Fertiliser", "Irrigation"]:
            if not self.find(k):
                v = {
                    "$type": f"Models.{k}, Models",
                    "Name": k,
                    "ResourceName": k,
                    "Children": [],
                    "Enabled": True,
                    "ReadOnly": False,
                }
                field["Children"].append(v)


# connect -> ok
# paused -> resume/get/set
# finished -> ok
class ApsimXEngine(BaseModelEngine):
    r"""Class for managing communication with an APSIMX server running
    in another process.

    Args:
        model_file: Path to a .apsimx model input file.
        apsimx_dir: Path to the directory containing APSIMX installation.
        **kwargs: Additional keyword arguments are passed to the
            BaseModelEngine constructor.

    """

    STATUS_MESSAGES = [
        "connect", "finished", "error", "recoverable_error",
    ]
    ERROR_MESSAGES = [
        "error", "recoverable_error",
    ]
    INPUT_FILE_TYPE = ApsimXFile
    AVAILABLE_ACTION_MAP = {
        "sow": {
            "description": "Sow a crop",
            "param": [
                "cropName", "cultivarName", "population",
                "sowingDepth", "rowSpacing",
            ],
            "num_levels": 1,  # Boolean
        },
        "harvest": {
            "description": "Harvest a crop",
            "param": [
                "cropName",
            ],
            "num_levels": 1,  # Boolean
        },
        "tillage": {
            "description": "Till the field",
            "param": ["type"],
            "num_levels": 1,  # Boolean
        },
        "nitrogen": {
            "alias": "N",
            "max": 8.0,
            "units": "kg/ha",
            "description": "Apply {amount} {units} nitrogen fertilizer.",
            # "cost": 0.46,  # $/kg
            "param": ["amount", "type", "depth", "depthBottom"],
        },
        "irrigate": {
            "alias": "water",
            "max": 20.0,
            "units": "mm",
            "description": "Irrigate with {amount} {units} of water.",
            # 1 mm over 1 ha == 10000 L, $20 per 10000 L
            # "cost": 20.0,  # $/mm/ha
            "param": ["amount"],
        },
    }

    def __init__(
            self,
            model_file: str,
            apsimx_dir: Optional[str] = None,
            **kwargs
    ):
        if apsimx_dir is None:
            apsimx_dir = _apsimxdir
        if not (isinstance(apsimx_dir, str) and os.path.isdir(apsimx_dir)):
            raise RuntimeError(f"APSIMX directory does not "
                               f"exist: \"{apsimx_dir}\"")
        self.apsimx_dir = apsimx_dir
        self.apsim_srv = os.path.join(
            self.apsimx_dir, "bin", "Debug", "net8.0",
            "ApsimZMQServer.dll")
        self.context = None
        self.socket = None
        self.port = None
        self.process = None
        self.stdout_pipe = None
        self.stderr_pipe = None
        self._status = None
        self._current_time = None
        if not os.path.isfile(self.apsim_srv):
            raise RuntimeError(f"APSIMX server executable does not "
                               f"exist: \"{self.apsim_srv}\"")
        super().__init__(model_file, **kwargs)
        if not self.output_dir:
            # ApsimX saves output to the directory containing the
            # model input file
            self.output_dir = os.path.dirname(self.model.fname)

    def update_model_file(self):
        r"""Update the model file to make it interactive and set the
        start/end times."""
        if self.sow_date is not None:
            self.action_map.pop("sow", None)
        if self.harvest_date is not None:
            self.action_map.pop("harvest", None)
        if self.harvest_date or "harvest" in self.action_map:
            self.model.disable("Harvest")
        if self.sow_date or "sow" in self.action_map:
            self.model.disable("Sow using a variable rule")
        # TODO: Add other actions?
        super().update_model_file()

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

    def start(self):
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
        assert self.status == "paused"
        if self.start_time is None:
            self.start_time = self.get("[Clock].Start")
        else:
            assert self.get("[Clock].Start") == self.start_time
        if self.end_time is None:
            self.end_time = self.get("[Clock].End")
        else:
            assert self.get("[Clock].End") == self.end_time
        logger.info(f"Simulating from {self.start_time} to {self.end_time}")

    def stop(self):
        r"""Stop the listening server and close the communication port."""
        if self.is_operable:
            self.act("terminate")
            if self.status != "finished":
                raise ValueError(
                    f"Status after terminate is \"{self.status}\"")
            self._status = "terminated"
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
        if isinstance(value, datetime.datetime):
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


class ApsimXEnv(BaseModelEnv):
    r"""ApsimX environment."""

    MODEL_ENGINE_CLASS = ApsimXEngine
    DEFAULT_ACTIONS = ["nitrogen", "irrigate"]
    DEFAULT_REVENUE_VAR = {
        "name": "[Wheat].Grain.Total.Wt",
        # "name": "Yield",  # Only includes harvested weight
        # "cost": ??,  # $/kg/ha
    }
