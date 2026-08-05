import os
import threading
import logging
from typing import Optional, Union, Any
from io import BufferedReader
from . import logger


_gymdir = os.path.dirname(__file__)
_apsimxdir = os.path.dirname(os.path.dirname(_gymdir))
if os.path.basename(_apsimxdir) != 'ApsimX':
    _apsimxdir = None
_datadir = os.path.join(_gymdir, "data")


class LogPipe(threading.Thread):
    r"""Thread to move output from a process PIPE to the logger.

    Args:
        pipe: Pipe that output should be streamed from.
        level: Integer logging level or the name of the logging level.
        prefix: Prefix to add to log messages.
        daemon: True if thread should be daemon.
        **kwargs: Additional keyword arguments are passed to the
            threading.Thread constructor.

    """

    def __init__(self, pipe: BufferedReader,
                 level: Optional[Union[str, int]] = "INFO",
                 prefix: Optional[str] = "",
                 daemon: Optional[bool] = True,
                 **kwargs: Any) -> None:
        r"""Initialize the LogPipe thread.

        Args:
            pipe: Pipe that output should be streamed from.
            level: Integer logging level or the name of the logging
                level.
            prefix: Prefix to add to log messages.
            daemon: True if thread should be daemon.
            **kwargs: Additional keyword arguments are passed to the
                threading.Thread constructor.

        """
        self.level = level
        self.prefix = prefix
        if isinstance(level, str):
            self.level = getattr(logging, level)
        self.pipe = pipe
        self.terminated = threading.Event()
        super(LogPipe, self).__init__(daemon=daemon, **kwargs)
        self.start()

    def close(self) -> None:
        r"""Close the pipe."""
        self.terminated.set()
        self.pipe.close()
        self.join()

    def run(self) -> None:
        r"""Run the thread, moving messages from the pipe to the
        logger."""
        for line0 in iter(self.pipe.readline, ''):
            line = line0.decode().strip('\n')
            if line:
                logger.log(self.level, self.prefix + line)
            if self.terminated.is_set():
                break
        self.terminated.set()
