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


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="action", help='Action to perform')
    # For running
    parser_run = subparsers.add_parser(
        "run", help="Run a simulation"
    )
    parser_run.add_argument(
        "model", type=str,
        help="Path to a .apsimx model input file",
    )
    # For creating interactive .apsimx
    parser_apsimx = subparsers.add_parser(
        "apsimx", help="Create a .apsimx model input file"
        # interactive version of a .apsimx model"
    )
    parser_apsimx.add_argument(
        "crop_name", type=str,  # choices=
        help="Path to a .apsimx model input file",
    )
    parser_apsimx.add_argument(
        "--from-example", type=str, nargs="?", const=True, default=False,
        help=(
            "Create a new .apsimx model by copying an example. "
            "The path to the example can be passed."
        ),
    )
    parser_apsimx.add_argument(
        "--interactive", action="store_true",
        help="Make the new file interactive",
    )
    parser_apsimx.add_argument(
        "--actions", type=str, nargs="+", action="extend",
        help="Interactive actions that should be supported",
    )
    parser_apsimx.add_argument(
        "--dst", type=str,
        help="Path to where the new .apsimx file should be saved",
    )
    parser_apsimx.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite any existing file",
    )
    # Generic arguments
    for x_parser in [parser_run, parser_apsimx]:
        x_parser.add_argument(
            "--apsimx-dir", type=str,
            help=(
                "Path to the root directory containing a APSIMX "
                "installation (i.e. the directory that contains "
                "\"bin/Debug/net8.0/ApsimZMQServer.dll\""
            ),
            default=_apsimxdir,
        )
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
        # TODO: Generic
        out = ApsimXFile.from_crop_name(
            args.crop_name,
            dst=args.dst,
            from_example=args.from_example,
            interactive=args.interactive,
            actions=args.actions,
            model_dir=args.apsimx_dir
        )
        out.write(overwrite=args.overwrite)
        out.generated = False  # Prevent cleanup
        print(f"Created input file \"{out.fname}\"")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
