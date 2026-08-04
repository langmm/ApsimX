import os
import glob
import json
import numpy as np
import pandas as pd
import datetime
from abc import abstractmethod
from typing import Optional, Union, List, Any
from .base import (
    readonly_cached_property, NoDefault,
    BaseModelFile, BaseModelEngine,
    BaseModelLLMPromptGenerator, BaseModelEnv,
)
from . import logger


class CropModelFile(BaseModelFile):
    r"""Base class for managing crop model input files."""

    @classmethod
    @abstractmethod
    def crop2fname(cls, crop_name: str,
                   model_dir: Optional[str] = None) -> str:
        r"""Locate an input model file for a given crop name.

        Args:
            crop_name: Crop name.
            model_dir: Directory containing the model.

        Returns:
            str: Model input file for the specified crop.

        """
        raise NotImplementedError  # pragma: no cover

    @classmethod
    @abstractmethod
    def available_crops(cls, model_dir: Optional[str] = None) -> List[str]:
        r"""Get the crops that can be simulated via this model.

        Args:
            model_dir: Directory containing the model.

        Returns:
            list: Available crop names.

        """
        raise NotImplementedError  # pragma: no cover

    @classmethod
    @abstractmethod
    def available_cultivars(cls, crop_name: str,
                            model_dir: Optional[str] = None) -> List[str]:
        r"""Get the cultivars for a given crop that can be simulated
        via this model.

        Args:
            crop_name: Crop name.
            model_dir: Directory containing the model.

        Returns:
            list: Available crop cultivar names.

        """
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def validate_crop_name(cls, crop_name: str,
                           model_dir: Optional[str] = None) -> str:
        r"""Ensure the crop name is one of those that can be simulated,
        normalizing it if necessary.
        Args:
            crop_name: Crop name.
            model_dir: Directory containing the model.

        Returns:
            str: Normalized crop name.

        """
        available_crops = cls.available_crops(model_dir=model_dir)
        for alias in [crop_name, crop_name.lower(), crop_name.title()]:
            if alias in available_crops:
                return alias
        raise NotImplementedError(
            f"Invalid crop name \"{crop_name}\". "
            f"Valid crop names are:\n\t"
            + "\n\t".join(available_crops))

    @classmethod
    @abstractmethod
    def from_crop_name(cls, crop_name: str,
                       model_dir: Optional[str] = None) -> "CropModelFile":
        r"""Create an input model file for a given crop name.

        Args:
            crop_name: Crop name.
            model_dir: Directory where the generated model should be saved.

        Returns:
            CropModelFile: Constructed model input file.

        """
        raise NotImplementedError  # pragma: no cover

    @BaseModelFile.parameter_property
    def crop_name(self):
        r"""str: Crop name."""
        return os.path.splitext(
            os.path.basename(self.fname_orig))[0].lower()

    @BaseModelFile.parameter_property
    def crop_variety(self):
        r"""str: Crop cultivar name."""
        return None  # raise KeyError("crop_variety")

    @BaseModelFile.parameter_property
    def location(self):
        r"""str: Description of the field location."""
        try:
            lat = self.get("latitude")
            lon = self.get("longitude")
            return f"{lat}°N, {lon}°E"
        except KeyError:
            return "the field"

    @BaseModelFile.parameter_property
    def field_area(self):
        r"""float: Field area"""
        return 1.0


class BaseWeatherFile(BaseModelFile):
    r"""Base class for weather files."""

    CACHED = True
    REQUIRED_POWER_PARAMETERS = [
        "TOA_SW_DWN",
        "ALLSKY_SFC_SW_DWN",  # MJ
        "T2M", "T2M_MIN", "T2M_MAX",  # C
        "T2MDEW",  # C
        "WS2M",  # wind
        "PRECTOTCORR",  # mm
    ]
    _default_ext = ".json"
    _default_start_date = datetime.date(1981, 1, 1)
    _default_end_date = datetime.date(2026, 5, 8)
    _default_cache_dir = os.path.join(
        os.getcwd(), "nasa_power_weather_data")
    _min_start_date = datetime.date(1981, 1, 1)
    _max_start_date = datetime.date.today()
    _make_interactive = None

    @property
    def parameters(self) -> list:
        r"""list: Set of power parameters contained by this file."""
        return self.REQUIRED_POWER_PARAMETERS

    @property
    @abstractmethod
    def dates(self) -> np.ndarray:
        r"""np.ndarray: Dates covered by this file."""
        raise NotImplementedError  # pragma: no cover

    @property
    @abstractmethod
    def latitude(self) -> float:
        r"""float: Latitude (degrees)."""
        raise NotImplementedError  # pragma: no cover

    @property
    @abstractmethod
    def longitude(self) -> float:
        r"""float: Longitude (degrees)."""
        raise NotImplementedError  # pragma: no cover

    @property
    def start_date(self) -> datetime.date:
        r"""datetime.date: Minimum date covered by this file."""
        return min(self.dates)

    @property
    def end_date(self) -> datetime.date:
        r"""datetime.date: Maximum date covered by this file."""
        return max(self.dates)

    @classmethod
    def fetch_data(cls, *args, **kwargs):
        r"""Look for an existing file that contains the data for the
        requested location and dates. If one does not exist, create it
        by downloading data from NASA POWER and converting it to the
        correct format.

        Args:
            *args, **kwargs: Arguments are passed along to from_location.

        Returns:
            str: File name.

        """
        instance = cls.from_location(*args, **kwargs)
        if not instance.exists:
            instance.write()
        return instance.fname

    @classmethod
    def from_location(cls, *args, **kwargs) -> "BaseWeatherFile":
        r"""Create a weather file from a location by requesting NASA
        power weather data.

        Args:
            *args, **kwargs: Arguments are passed to
                NASAPOWERWeatherFile.from_location.

        Returns:
            BaseWeatherFile: File instance.

        """
        fpower = NASAPOWERWeatherFile.from_location(*args, **kwargs)
        return cls.from_power(fpower)

    @classmethod
    def from_power(cls, fpower: Union[str, "NASAPOWERWeatherFile"],
                   fname: Optional[str] = None) -> "BaseWeatherFile":
        r"""Create a weather file from NASA power weather data.

        Args:
            src: JSON file containing NASA POWER data.
            fname: File name where the weather data should be written to.

        """
        if isinstance(fpower, str):
            fpower = NASAPOWERWeatherFile(fpower)
        if fname is None:
            fname = os.path.splitext(fpower.fname)[0] + cls._default_ext
        if os.path.isfile(fname):
            return cls(fname)
        fpower.add_missing_param(cls.REQUIRED_POWER_PARAMETERS)
        contents = cls._from_power(fpower.contents)
        return cls(fname, contents=contents)

    def covers_range(self,
                     start: Union[datetime.date, datetime.datetime],
                     end: Union[datetime.date, datetime.datetime],
                     latitude: Optional[float] = None,
                     longitude: Optional[float] = None):
        r"""Check if the file contains data for the specified date/time
        range.

        Args:
            start: Start of range.
            end: End of range.
            latitude: Latitude that data should cover.
            longitude: Longitude that data should cover.

        Returns:
            bool: True if the range is covered, False otherwise.

        """
        if isinstance(start, datetime.datetime):
            start = start.date()
        if isinstance(end, datetime.datetime):
            end = end.date()
        start_date = self.start_date
        end_date = self.end_date
        if latitude is not None and self.latitude != latitude:
            return False
        if longitude is not None and self.longitude != longitude:
            return False
        return (start >= start_date and start < end_date
                and end > start_date and end <= end_date)

    @classmethod
    @abstractmethod
    def _from_power(cls, src: dict):
        r"""Convert NASA power data into the correct format for this
        file.

        Args:
            src: NASA power data.

        Returns:
            Converted data.

        """
        raise NotImplementedError  # pragma: no cover


class NASAPOWERWeatherFile(BaseWeatherFile):
    r"""Wrapper for loading NASA POWER data."""

    CACHED = True
    _default_ext = ".json"

    @readonly_cached_property
    def parameters(self) -> list:
        r"""list: Set of power parameters contained by this file."""
        return list(self.contents["properties"]["parameter"].keys())

    @readonly_cached_property
    def dates(self) -> np.ndarray:
        r"""np.ndarray: Dates covered by this file."""
        dates = pd.Series(
            self.contents["properties"]["parameter"][self.parameters[0]])
        return pd.to_datetime(dates.index, format="%Y%m%d")

    @readonly_cached_property
    def latitude(self) -> float:
        r"""float: Latitude (degrees)."""
        return float(self.contents["geometry"]["coordinates"][0])

    @readonly_cached_property
    def longitude(self) -> float:
        r"""float: Longitude (degrees)."""
        return float(self.contents["geometry"]["coordinates"][1])

    @readonly_cached_property
    def start_date(self) -> datetime.date:
        r"""datetime.date: Start of range covered by the file."""
        return datetime.datetime.strptime(
            self.contents["header"]["start"], "%Y%m%d").date()

    @readonly_cached_property
    def end_date(self) -> datetime.date:
        r"""datetime.date: End of range covered by the file."""
        return datetime.datetime.strptime(
            self.contents["header"]["end"], "%Y%m%d").date()

    def add_missing_param(self, parameters: List[str]):
        r"""Fill in any missing parameters.

        Args:
            parameters: Set of parameters that must be present.

        """
        missing = [k for k in parameters if k not in self.parameters]
        if not missing:
            return
        data = self.download_data(self.latitude, self.longitude,
                                  self.start_date, self.end_date,
                                  parameters=missing)
        self.contents["properties"]["parameter"].update(
            data["properties"]["parameter"])
        self.write(overwrite=True)

    @classmethod
    def _read(cls, fname: str):
        r"""Read a model input file.

        Args:
            fname: Path to file to read.

        Returns:
            object: File contents.

        """
        with open(fname, "r") as fd:
            return json.load(fd)

    @classmethod
    def _write(cls, fname: str, contents):
        r"""Read a model input file.

        Args:
            fname: Path to file to read.
            contents: File contents to write.

        """
        with open(fname, "w") as fd:
            json.dump(contents, fd)

    @classmethod
    def _from_power(cls, src: dict):
        r"""Convert NASA power data into the correct format for this
        file.

        Args:
            src: NASA power data.

        Returns:
            Converted data.

        """
        return src

    @classmethod
    def format_filename(cls, latitude: float, longitude: float,
                        start_date: Optional[datetime.date] = None,
                        end_date: Optional[datetime.date] = None,
                        cache_dir: Optional[str] = None) -> str:
        r"""Construct the file name for the cached NASA power file
        containing the requested data.

        Args:
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            start_date: Starting date for data.
            end_date: Ending date for data.
            cache_dir: Directory where the data should be cached.

        Returns:
            str: File name.

        """
        start_date = start_date or cls._default_start_date
        end_date = end_date or cls._default_end_date
        cache_dir = cache_dir or cls._default_cache_dir

        def f2str(x):
            if isinstance(x, datetime.date):
                return x.isoformat()
            elif isinstance(x, float):
                return str(x).replace(".", "p").replace("-", "n")
            return x

        return os.path.join(
            cache_dir,
            f"{f2str(latitude)}_{f2str(longitude)}_"
            f"{f2str(start_date)}_to_{f2str(end_date)}.json"
        )

    @classmethod
    def from_location(cls, latitude: float, longitude: float,
                      start_date: Union[datetime.date, datetime.datetime],
                      end_date: Union[datetime.date, datetime.datetime],
                      parameters: Optional[List[str]] = None,
                      cache_dir: Optional[str] = None):
        r"""Look for an existing file that contains the data for the
        requested location and dates. If one does not exist, create it
        by downloading data from NASA POWER.

        Args:
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            start_date: Starting date for data.
            end_date: Ending date for data.
            parameters: Set of parameters that should be included in
                the data.
            cache_dir: Directory where the data should be cached.

        Returns:
            NASAPOWERWeatherFile: File instance.

        """
        if isinstance(start_date, datetime.datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime.datetime):
            end_date = end_date.date()
        fglob = cls.format_filename(latitude, longitude, "*", "*",
                                    cache_dir=cache_dir)
        for f in glob.glob(fglob):
            fparts = os.path.splitext(os.path.basename(f))[0].split("_")
            fstart = datetime.date.fromisoformat(fparts[-3])
            fend = datetime.date.fromisoformat(fparts[-1])
            if fstart <= start_date and fend >= end_date:
                out = cls(f)
                if parameters:
                    out.add_missing_param(parameters)
                return out
        start_date = min(start_date, cls._default_start_date)
        end_date = max(end_date, cls._default_end_date)
        assert end_date > start_date
        if start_date < cls._min_start_date:
            raise ValueError(
                f"The requested start date ({start_date}) predates the "
                f"minimum ({cls._min_start_date})")
        if end_date > cls._max_start_date:
            raise ValueError(
                f"The requested end date ({end_date}) excedes the "
                f"maximum ({cls._max_start_date})")
        fname = cls.format_filename(latitude, longitude,
                                    start_date, end_date,
                                    cache_dir=cache_dir)
        assert not os.path.isfile(fname)
        data = cls.download_data(latitude, longitude,
                                 start_date, end_date,
                                 parameters=parameters)
        if not os.path.isdir(os.path.dirname(fname)):
            os.mkdir(os.path.dirname(fname))
        out = cls(fname, contents=data)
        out.write()
        return out

    @classmethod
    def download_data(cls, latitude: float, longitude: float,
                      start_date: Optional[datetime.date],
                      end_date: Optional[datetime.date],
                      parameters: Optional[List[str]] = None) -> dict:
        r"""Use REST API to get NASA POWER data for a location.

        Args:
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            start_date: Starting date for data.
            end_date: Ending date for data.
            parameters: Set of parameters to request.

        Returns:
            dict: JSON result.

        """
        # Based on PCSE _query_NASAPower_server
        import requests
        if parameters is None:
            parameters = cls.REQUIRED_POWER_PARAMETERS
        # build URL for retrieving data, using new NASA POWER api
        server = "https://power.larc.nasa.gov/api/temporal/daily/point"
        payload = {
            "request": "execute",
            "parameters": ",".join(parameters),
            "latitude": latitude,
            "longitude": longitude,
            "start": start_date.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"),
            "community": "AG",
            "format": "JSON",
        }
        logger.debug("Starting retrieval from NASA POWER")
        req = requests.get(server, params=payload)
        if req.status_code != 200:
            raise RuntimeError(
                f"Failed retrieving POWER data, server returned HTTP "
                f"code: {req.status_code} on following URL {req.url}"
            )
        logger.debug("Successfully retrieved data from NASA POWER")
        return req.json()


class CropModelEngine(BaseModelEngine):
    r"""Class for managining communication with a crop simulation model.

    Args:
        model_file: Path to one or more model input files.
        crop_name: Name of the crop.
        crop_variety: Name of the crop variety/cultivar.
        sow_date: Date that the crop should be sown.
        harvest_date: Date that the crop should be harvested.
        season_length: Time between sowing and harvest. Only used if
            only one of sow_date or harvest_date are used. If an integer
            is provided, it is assumed to be in units of days.
        year: Year to use to get weather data.
        latitude: Field latitude to use to get weather data.
        longitude: Field longitude to use to get weather data.
        weather_file: Path to a file containing NASA power weather data.
        nasa_power_cache_dir: Directory where NASA POWER weather files
            should be cached.
        **kwargs: Additional keywords arguments are passed along to
            BaseModelEngine.__init__.

    """

    EXPLICIT_PARAM = BaseModelEngine.EXPLICIT_PARAM + [
        "crop_name", "crop_variety",
        "sow_date", "harvest_date", "season_length",
        "latitude", "longitude", "year",
        "weather_file",
    ]
    WEATHER_FILE_TYPE = None
    DATE_PARAM = BaseModelEngine.DATE_PARAM + [
        ("sow_date", "harvest_date", "season_length"),
    ]
    DEFAULT_PARAM = {
        "duration": datetime.timedelta(365),
    }

    def __init__(
            self,
            model_file: Optional[Union[str, List[str],
                                       BaseModelFile]] = None,
            crop_name: Optional[str] = None,
            crop_variety: Optional[str] = None,
            sow_date: Optional[datetime.date] = None,
            harvest_date: Optional[datetime.date] = None,
            season_length: Optional[Union[int, datetime.timedelta]] = None,
            year: Optional[int] = None,
            latitude: Optional[float] = None,
            longitude: Optional[float] = None,
            weather_file: Optional[str] = None,
            nasa_power_cache_dir: Optional[str] = None,
            **kwargs
    ):
        if isinstance(season_length, int):
            season_length = datetime.timedelta(season_length)
        self.crop_name = crop_name
        self.crop_variety = crop_variety
        self.sow_date = sow_date
        self.harvest_date = harvest_date
        self.season_length = season_length
        self.year = year
        self.latitude = latitude
        self.longitude = longitude
        self.weather_file = weather_file
        self.nasa_power_cache_dir = nasa_power_cache_dir
        if model_file is None:
            if not self.crop_name:
                raise ValueError("Either a model file or crop name must "
                                 "be provided")
            model_file = self.INPUT_FILE_TYPE.crop2fname(
                self.crop_name, model_dir=kwargs.get("model_dir", None))
        super().__init__(model_file, **kwargs)

    def update_model_file(self):
        r"""Update the model file to make it interactive and set the
        start/end times."""
        if self.latitude or self.longitude:
            assert self.latitude and self.longitude
        self.sync_param(["duration"], required=True, skip_file=True,
                        dont_update=True)
        self.sync_param(["start_time", "end_time", "year"],
                        dont_update=True, required=True)
        self.sync_param("weather_file", dont_update=True)
        download_weather_file = False
        if self.weather_file and not os.path.isfile(self.weather_file):
            logger.info(
                f"The specified weather file (\"{self.weather_file}\") "
                f"does not exist. Weather data will be downloaded from "
                f"NASA POWER."
            )
            self.del_param("weather_file")
            download_weather_file = True
        elif self.weather_file:
            weather_file = self.WEATHER_FILE_TYPE(self.weather_file)
            latitude = self.get_param("latitude", None, skip_file=True)
            longitude = self.get_param("longitude", None, skip_file=True)
            if not weather_file.covers_range(
                    self.start_time, self.end_time,
                    latitude=latitude, longitude=longitude,
            ):
                logger.info(
                    f"The provided weather file (valid for "
                    f"{weather_file.start_date} to "
                    f"{weather_file.end_date}, "
                    f"latitude={weather_file.latitude}, "
                    f"longitude={weather_file.longitude}) does not "
                    f"cover the simulation range ({self.start_time} to "
                    f"{self.end_time}, latitude={latitude}, "
                    f"longitude={longitude}). "
                    f"Weather data will be downloaded from NASA POWER."
                )
                self.del_param("weather_file")
                download_weather_file = True
        elif (self.has_param("latitude", skip_file=True)
              or self.has_param("longitude", skip_file=True)):
            download_weather_file = True
        if download_weather_file:
            self.sync_param(["latitude", "longitude"],
                            dont_update=True, required=True)
            self.weather_file = self.WEATHER_FILE_TYPE.fetch_data(
                self.latitude, self.longitude,
                self.start_time, self.end_time,
                cache_dir=self.nasa_power_cache_dir,
            )
            logger.info(
                f"Downloaded weather data: \"{self.weather_file}\"")
        # TODO: Soil file?
        if self.sow_date is not None:
            self.action_map.pop("sow", None)
        if self.harvest_date is not None:
            self.action_map.pop("harvest", None)
        super().update_model_file()

    def calc_param(self, name: str, default: Optional[Any] = NoDefault,
                   **kwargs):
        r"""Calculate a parameter from other parameters.

        Args:
            name: Parameter to calculate.
            default: Default to return if the parameter cannot be
                calculated.
            **kwargs: Additional keyword arguments are passed to
                parent class's calc_param.

        Returns:
            Calculated parameter value.

        """
        if name == "year":
            for k, _, _ in self.DATE_PARAM:
                try:
                    v = self.get_param(k, **kwargs)
                    return v.year
                except KeyError:
                    pass
        return super().calc_param(name, default=default, **kwargs)

    @property
    def location(self):
        r"""str: Description of the field location."""
        return self.model.location

    @property
    def field_area(self):
        r"""float: Field area"""
        return self.model.field_area


class CropModelLLMPromptGenerator(BaseModelLLMPromptGenerator):
    r"""Crop model LLM prompt generator."""

    DEFAULT_REWARD = (
        "Maximize the end-of-season profit from the crop yield"
    )
    DEFAULT_STATE_DESCRIPTOR = "agronomic"

    def __init__(
            self,
            crop_name: Optional[str] = "the crop",
            crop_variety: Optional[str] = None,
            start_date: Optional[datetime.date] = None,
            season_length: Optional[int] = 241,
            location: Optional[str] = "the field",
            **kwargs
    ):
        """Initialize the prompt generator.

        Args:
            crop_name: Name of the crop being cultivated
            crop_variety: Name of the crop varient/cultivar being
                cultivated
            start_date: Calendar start date of the simulation
            season_length: Total length of growing season in days
            location: Geographic location description
            **kwargs: Additional keyword arguments are forwarded to the
                BaseModelLLMPromptGenerator.__init__ method.

        """
        self.crop_name = crop_name
        self.crop_variety = crop_variety
        self.start_date = start_date
        self.season_length = season_length
        self.location = location
        super().__init__(**kwargs)

    @classmethod
    def from_env(
            cls,
            env: "CropModelEnv",
            **kwargs
    ) -> "CropModelLLMPromptGenerator":
        """Create prompt generator from a model gym environment.

        Args:
            env: model gym environment instance
            **kwargs: Additional keyword arguments are passed to the
                class constructor.

        Returns:
            CropModelLLMPromptGenerator instance configured for the
                environment.

        """
        return super().from_env(
            env,
            crop_name=env.model.crop_name,
            crop_variety=env.model.crop_variety,
            start_date=env.start_time,
            season_length=env.model.season_length,
            location=env.model.location,
            **kwargs
        )

    @property
    def crop_description(self) -> str:
        r"""str: Crop description including provided info."""
        out = self.crop_name
        if self.crop_variety:
            out = f"{self.crop_variety} {out}"
        return out

    def get_system_prompt(self) -> str:
        """Generate the system prompt for the LLM agent.

        Returns:
            System prompt string
        """
        # TODO: Include action costs
        base_prompt = (
            f"You are an agricultural management expert whose goal is "
            f"to {self.reward_inline} "
            f"throughout the entire growing season by selecting the "
            f"appropriate agricultural management action at each "
            f"step."
        )
        return base_prompt

    def turn_context(self, observation: np.ndarray) -> str:
        r"""Generate a string to summarize the current turn at a high
        level with the context of the simulation.

        Args:
            observation: Current state.

        Returns:
            str: Turn summary.

        """
        day_val = None
        for i, var in enumerate(self.output_vars):
            section, description = self.desc_map.get(var, ('Other', var))
            if ((section == "Timeline"
                 and description.upper().startswith("DAYS"))):
                day_val = observation[i]
                break
        day_num = day_val if day_val is not None else 0

        calendar_phrase = ""
        if self.start_date is not None and day_val is not None:
            try:
                calendar_date = (
                    self.start_date
                    + datetime.timedelta(
                        days=max(int(round(float(day_val))) - 1, 0))
                )
                calendar_phrase = (
                    f", corresponding to {calendar_date.strftime('%B')} "
                    f"{calendar_date.day}"
                )
            except Exception:
                pass

        intro = (
            f"We are growing {self.crop_description} from sowing to "
            f"maturity. The planned growing window spans "
            f"{self.season_length} days, actions are taken every "
            f"{self.intervention_interval} days, "
            f"and today is day {day_num:.0f} of the "
            f"season{calendar_phrase}."
        )
        return intro


class CropModelEnv(BaseModelEnv):
    r"""Crop model environment."""

    def get_output_vars(self) -> List[str]:
        r"""Get the output variables specified by the model file.

        Returns:
            list: Output variables

        """
        out = super().get_output_vars()
        return out + ["days_elapsed"]

    def _get_cost(self, action: dict) -> float:
        r"""Calculate the cost of the current action.

        Args:
            action: Current action.

        Returns:
            float: Action cost.

        """
        return self.model.field_area * super()._get_cost(action)

    def _get_revenue(self, obs: dict) -> float:
        r"""Calculate the revenue based on the current observation.

        Args:
            obs: Current observation.

        Returns:
            float: Revenue.

        """
        return self.model.field_area * super()._get_revenue(obs)

    def _process_observation(self, observation: dict):
        r"""Force the observations into the expected format.

        Args:
            observation: Raw observations.

        Returns:
            np.ndarray: Array of observations parameters.

        """
        observation = super()._process_observation(observation)
        days_elapsed = self.model.current_time - self.model.start_time
        return np.concatenate([
            observation,
            [days_elapsed.days],
        ])
