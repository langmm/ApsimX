# python RL-Gym/run_interactive.py run Examples/Wheat.apsimx
import os
import copy
import json
import zmq
import msgpack
import subprocess
import contextlib
import logging
import datetime
import threading
from functools import cached_property
logger = logging.getLogger(__name__)


_rootdir = os.path.dirname(os.path.dirname(__file__))
_syncfile = os.path.join(_rootdir, "RL-Gym", "Synchroniser.apsimx")


class LogPipe(threading.Thread):
    r"""Thread to move output from a process PIPE to the logger.

    Args:
        level (int, str): Integer logging level or the name of the
            logging level.

    """

    def __init__(self, pipe, level="INFO", **kwargs):
        self.level = level
        if isinstance(level, str):
            self.level = getattr(logging, level)
        self.pipe = pipe
        self.terminated = threading.Event()
        super(LogPipe, self).__init__(**kwargs)
        self.start()

    def fileno(self):
        """Return the write file descriptor of the pipe"""
        return self.fd_write

    def run(self):
        """Run the thread, moving messages from the pipe to the
        logger."""
        for line0 in iter(self.pipe.readline, ''):
            line = line0.decode().strip('\n')
            if line:
                logger.log(self.level, line)
            if self.terminated.is_set():
                break
        self.terminated.set()


# connect -> ok
# paused -> resume/get/set
# finished -> ok
class ExternalApsim:
    r"""Class for managing communication with an APSIMX server running
    in another process.

    Args:
        model (str): Path to a .apsimx model input file.
        apsim_dir (str, optional): Path to the directory containing
            APSIMX installation.

    """

    def __init__(self, model, apsim_dir=None):
        if apsim_dir is None:
            apsim_dir = _rootdir
        self.model = ApsimXFile(model)
        if not self.model.is_interactive:
            self.model = self.model.make_interactive()
        self.apsim_dir = apsim_dir
        self.apsim_srv = os.path.join(
            apsim_dir, "bin", "Debug", "net8.0", "ApsimZMQServer.dll")
        self.context = None
        self.socket = None
        self.port = None
        self.process = None
        self._status = None
        self.start_time = None
        self.current_time = None
        if not os.path.isfile(self.apsim_srv):
            raise RuntimeError(f"APSIMX server executable does not "
                               f"exist: \"{self.apsim_srv}\"")
        self.start()

    @classmethod
    def _pipe_to_logger(cls, pipe):
        for line in iter(pipe.readline, b''):
            logger.info(line.decode())

    @property
    def status(self):
        r"""str: Current simulation status."""
        if self._status is None:
            self._status = self.socket.recv_string()
            if self._status == "paused":
                self.current_time = self.get("[Clock].Today").to_datetime()
                logger.debug(f"Simulation waiting at {self.current_time}")
            elif self._status in ["connect", "finished"]:
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
        # The simulation will connect back to this port:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
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
        self.stdout_pipe = LogPipe(self.process.stdout)
        self.stderr_pipe = LogPipe(self.process.stderr, level="ERROR")
        # with self.process.stdout:
        #     self._pipe_to_logger(self.process.stdout)
        logger.info(f"Started APSIMX process id: {self.process.pid}")
        assert self.status == "connect"
        assert self.status == "paused"
        self.start_time = self.current_time
        logger.info(f"Simulation start time: {self.start_time}")

    def stop(self):
        r"""Stop the listening server and close the communication port."""
        self.socket.close()
        self.process.terminate()
        self.stdout_pipe.terminated.set()
        self.stderr_pipe.terminated.set()
        # process = psutil.Process(self.process.pid)
        # for proc in process.children(recursive=True):
        #     proc.kill()
        # process.kill()

    def send_command(self, command, args=None):
        r"""Send a command to the server process, e.g. resume/set/get.

        Args:
            command (str): Command to send.
            args (list, optional): Additional arguments to send with the
                commaned.

        """
        assert self.status is not None
        self.socket.send_string(command, zmq.SNDMORE if args else 0)
        if args:
            for i, arg in enumerate(args):
                self.socket.send(msgpack.packb(arg),
                                 zmq.SNDMORE if i < len(args) - 1 else 0)
        if self.status != 'finished':
            self._status = None

    def check_paused(self):
        r"""Check that the simulation server is paused."""
        assert self.status == "paused"

    @contextlib.contextmanager
    def ensure_paused(self):
        r"""Context manager that ensures the simulation is paused."""
        self.check_paused()
        status = self.status
        yield
        self._status = status

    def get(self, name):
        r"""Send a request to get the current value of a simulation state
        variable.

        Args:
            name (str): Name of variable to get the value of.

        Returns:
            object: Current variable value.

        """
        out = None
        with self.ensure_paused():
            self.send_command("get", [name])
            out = msgpack.unpackb(self.socket.recv())
            logger.debug(f"{name} = {out}")
        return out

    def set(self, name, value):
        r"""Send a request to set a simulation state variable.

        Args:
            name (str): Name of the variable to update.
            value (object): New value for the named variable.

        """
        with self.ensure_paused():
            self.send_command("set", [name, value])
            assert self.socket.recv_string() == "ok"

    def do(self, action, **kwargs):
        r"""Send a request to perform a management action.

        Args:
            action (str): Name of action to perform. Supported actions
                include::

                     "addFertiliser": Apply fertilizer.
                     "sowCrop": Sow a crop.
                     "harvestCrop": Harvest a crop.
                     "applyIrrigation": Apply irrigation.
                     "tillage": Till the field.
                     "terminate": Stop the simulation.

            **kwargs: Additional keyword arguments are packaged as the
                actions parameters.

        """
        allowed_param = {
            "addFertiliser": [
                "amount", "type"],
            "sowCrop": [
                "cropName", "cultivarName", "population",
                "sowingDepth", "rowSpacing"],
            "harvestCrop": [
                "cropName"],
            "applyIrrigation": [
                "amount"],
            "tillage": [
                "type"],
            "terminate": [],
        }
        if action not in allowed_param:
            raise ValueError(
                f"Unsupported action \"{action}\". Supported "
                f"actions are: {list(allowed_param.keys())}")
        args = []
        for k in allowed_param[action]:
            if k in kwargs:
                args += [k, kwargs.pop(k)]
        if kwargs:
            raise ValueError(
                f"Unsupported parameters were provided for "
                f"action \"{action}\": {list(kwargs.keys())}")
        with self.ensure_paused():
            self.send_command("do", [action] + args)
            assert self.socket.recv_string() == "ok"

    def getvars(self, names):
        r"""Send a request to get the current value of a set of
        simulation state variables.

        Args:
            names (list): Names of variables to get values for.

        Returns:
            dict: Mapping between state variable names and retrieved
                values.

        """
        out = {}
        for name in names:
            out[name] = self.get(name)
        return out

    def setvars(self, values):
        r"""Send a request to set simulation state variables.

        Args:
            values (dict): Mapping between state variable names and the
                values they should be set to.

        """
        for k, v in values.items():
            self.set(k, v)

    def resume(self, until=None):
        r"""Resume the simulation.

        Args:
            until (datetime, optional): Time that simulation should be
                run to.

        """
        self.check_paused()
        self.send_command("resume")
        if until is not None:
            while (self.status != "finished"
                   and self.current_time < until):
                self.resume()


class ApsimXFile:
    r"""Container for manipulating .apsimx model files.

    Args:
        fname (str): Path to a .apsimx model file.

    """

    def __init__(self, fname):
        self.fname = fname

    @cached_property
    def contents(self):
        r"""dict: File contents."""
        with open(self.fname, 'r') as fd:
            return json.load(fd)

    @cached_property
    def is_interactive(self):
        r"""bool: True if the .apsimx model is interactive."""
        return bool(self.find("Synchroniser"))

    def write(self, new_contents=None):
        r"""Write a new set of contents to the file.

        Args:
            new_contents (dict, optional): New contents to write.

        """
        if new_contents is not None:
            self.contents = new_contents
            del self.is_interactive
        with open(self.fname, 'w') as fd:
            json.dump(self.contents, fd, indent="    ")

    def find(self, name, current=None, parent=False):
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
        return False

    def make_interactive(self, dst=None):
        r"""Create an interactive version of this .apsimx model.

        Args:
            dst (str, optional): Path to the location where the
                interactive .apsimx model should be saved.

        Returns:
            ApsimXFile: Interactive .apsimx model.

        """
        if dst is None:
            dst = '-Interactive'.join(os.path.splitext(self.fname))
        dst = ApsimXFile(dst)
        dst.contents = copy.deepcopy(self.contents)
        sync = ApsimXFile(_syncfile)
        if self.is_interactive:
            logger.warn(f"Source .apsimx \"{self.fname}\" is already "
                        f"interactive")
        else:
            field = dst.find("Field")
            assert field
            field["Children"].append(copy.deepcopy(sync.contents))
            for k in ["Fertiliser", "Irrigation"]:
                if not dst.find(k):
                    v = {
                        "$type": f"Models.{k}, Models",
                        "Name": k,
                        "ResourceName": k,
                        "Children": [],
                        "Enabled": True,
                        "ReadOnly": False,
                    }
                    field["Children"].append(v)
        dst.write()
        return dst

    def run(self):
        r"""Run this .apsimx model."""
        run(self.fname)


def run(model, **kwargs):
    r"""Run a simulation."""
    apsim = ExternalApsim(model, **kwargs)
    try:
        i = 0
        while apsim.status != 'finished':
            logger.info(f"Time: {apsim.current_time}")
            apsim.getvars([
                "[Wheat].Phenology.Zadok.Stage",
                "[Soil].Water.PAW",
            ])
            # Decision point
            if i % 2 == 0:
                apsim.do("addFertiliser", amount=160)  # kg/ha
            else:
                apsim.do("applyIrrigation", amount=10)  # mm
            # reply = apsim.get("[Nutrient].NO3.kgha")
            # new_value = [2*ele for ele in reply]
            # apsim.set("[Nutrient].NO3.kgha", new_value)
            # reply = apsim.get("[Nutrient].NO3.kgha")
            # assert reply == new_value
            next_time = apsim.current_time + datetime.timedelta(days=10)
            apsim.resume(until=next_time)
            i += 1
            # TODO: Add irrigation
    finally:
        apsim.stop()
    print(f"Output written to {apsim.output_file}")


def create_interactive_apsimx(src, dst=None):
    r"""Create an interactive version of a .apsimx model.

    Args:
        src (str, ApsimXFile): Path to the source .apsimx model.
        dst (str, optional): Path to the location where the generated
            interactive .apsimx model should be saved.

    """
    if not isinstance(src, ApsimXFile):
        src = ApsimXFile(src)
    dst = src.make_interactive(dst=dst)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="action", help='subcommand help')
    # For running
    parser_run = subparsers.add_parser(
        "run", help="Run a simulation"
    )
    parser_run.add_argument(
        "model", type=str,
        help="Path to a .apsimx model input file",
        default=os.path.join(_rootdir, "RL-Gym",
                             "Wheat-Interactive.apsimx"),
    )
    parser_run.add_argument(
        "--apsim-dir", type=str,
        help=(
            "Path to the root directory containing a APSIMX "
            "installation (i.e. the directory that contains "
            "\"bin/Debug/net8.0/ApsimZMQServer.dll\""
        ),
        default=_rootdir,
    )
    # For creating interactive .apsimx
    parser_apsimx = subparsers.add_parser(
        "apsimx", help="Create an interactive version of a .apsimx model"
    )
    parser_apsimx.add_argument(
        "model", type=str,
        help="Path to a .apsimx model input file",
    )
    parser_apsimx.add_argument(
        "--dst", type=str,
        help="Path to where the interactive .apsimx file should be saved"
    )
    # Generic arguments
    for x_parser in [parser_run, parser_apsimx]:
        x_parser.add_argument(
            "--log-file", type=str, nargs="?", const=True,
            help="File where log message should be written",
        )
        x_parser.add_argument(
            "--log-level", choices=[
                "NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            ],
            help="Logging level", default="INFO",
        )
    args = parser.parse_args()
    if args.log_file is True:
        args.log_file = os.path.join(
            os.path.splitext(args.model)[0] + ".log")
    logging.basicConfig(filename=args.log_file,
                        level=getattr(logging, args.log_level))
    if args.log_file:
        print(f"Log being written to \"{args.log_file}\"")
    if args.action == "run":
        run(args.model, apsim_dir=args.apsim_dir)
    elif args.action == "apsimx":
        if args.dst is None:
            args.dst = '-Interactive'.join(os.path.splitext(args.model))
        create_interactive_apsimx(args.model, args.dst)
