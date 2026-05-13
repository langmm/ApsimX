# python -m apsimx_gym.cli run Examples/Wheat.apsimx
import os
import sys
import argparse
import datetime
import logging
from . import logger
from .utils import _apsimxdir
from .engine import ApsimXFile, ApsimXEngine


def run(model, **kwargs):
    r"""Run a simulation."""
    apsim = ApsimXEngine(model, **kwargs)
    apsim.start()
    try:
        i = 0
        while apsim.is_running and not apsim.is_complete:
            logger.info(f"Time: {apsim.current_time}")
            apsim.getvars([
                "[Wheat].Phenology.Zadok.Stage",
                "[Soil].Water.PAW",
            ])
            # Decision point
            if i % 2 == 0:
                apsim.act("nitrogen", amount=160)  # kg/ha
            else:
                apsim.act("irrigate", amount=10)  # mm
            # reply = apsim.get("[Nutrient].NO3.kgha")
            # new_value = [2*ele for ele in reply]
            # apsim.set("[Nutrient].NO3.kgha", new_value)
            # reply = apsim.get("[Nutrient].NO3.kgha")
            # assert reply == new_value
            apsim.fast_forward(datetime.timedelta(days=10))
            i += 1
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
    dst = src.copy(dst=dst)
    actions = list(ApsimXEngine.AVAILABLE_ACTION_MAP.keys())
    dst.make_interactive(actions)
    dst.write()


def main():
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
    )
    parser_run.add_argument(
        "--apsimx-dir", type=str,
        help=(
            "Path to the root directory containing a APSIMX "
            "installation (i.e. the directory that contains "
            "\"bin/Debug/net8.0/ApsimZMQServer.dll\""
        ),
        default=_apsimxdir,
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
        run(args.model, model_dir=args.apsimx_dir)
    elif args.action == "apsimx":
        if args.dst is None:
            args.dst = '-Interactive'.join(os.path.splitext(args.model))
        create_interactive_apsimx(args.model, args.dst)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
