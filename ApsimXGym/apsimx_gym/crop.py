import os
import glob
import json
import numpy as np
import datetime
from abc import abstractmethod
from typing import Optional, Union, List
from .base import (
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

    _default_ext = ".json"
    _default_start_date = datetime.date(1981, 1, 1)
    _default_end_date = datetime.date(2026, 5, 8)
    _default_cache_dir = os.path.join(
        os.getcwd(), "nasa_power_weather_data")
    _min_start_date = datetime.date(1981, 1, 1)
    _max_start_date = datetime.date.today()

    @classmethod
    def format_NASAPower_file(cls, latitude: float, longitude: float,
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
    def get_NASAPower_file(cls, latitude: float, longitude: float,
                           start_date: datetime.date,
                           end_date: datetime.date,
                           cache_dir: Optional[str] = None):
        r"""Look for an existing file that contains the data for the
        requested location and dates. If one does not exist, create it.

        Args:
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            start_date: Starting date for data.
            end_date: Ending date for data.
            cache_dir: Directory where the data should be cached.

        Returns:
            str: File name.

        """
        fglob = cls.format_NASAPower_file(latitude, longitude,
                                          "*", "*",
                                          cache_dir=cache_dir)
        for f in glob.glob(fglob):
            fparts = os.path.splitext(os.path.basename(f))[0].split("_")
            fstart = datetime.date.fromisoformat(fparts[-3])
            fend = datetime.date.fromisoformat(fparts[-1])
            if fstart <= start_date and fend >= end_date:
                return f
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
        fname = cls.format_NASAPower_file(latitude, longitude,
                                          start_date, end_date,
                                          cache_dir=cache_dir)
        assert not os.path.isfile(fname)
        data = cls.fetch_NASAPower_data(latitude, longitude,
                                        start_date, end_date)
        if not os.path.isdir(os.path.dirname(fname)):
            os.mkdir(os.path.dirname(fname))
        with open(fname, "w") as fd:
            json.dump(data, fd)
        return fname

    @classmethod
    def fetch_NASAPower_data(cls, latitude: float, longitude: float,
                             start_date: Optional[datetime.date],
                             end_date: Optional[datetime.date]) -> dict:
        r"""Use REST API to get NASA Power data for a location.

        Args:
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            start_date: Starting date for data.
            end_date: Ending date for data.

        Returns:
            dict: JSON result.

        """
        # Based on PCSE _query_NASAPower_server
        import requests
        # build URL for retrieving data, using new NASA POWER api
        server = "https://power.larc.nasa.gov/api/temporal/daily/point"
        payload = {
            "request": "execute",
            "parameters": ",".join([
                "TOA_SW_DWN",
                "ALLSKY_SFC_SW_DWN",  # MJ
                "T2M", "T2M_MIN", "T2M_MAX",  # C
                "T2MDEW",  # C
                "WS2M",  # wind
                "PRECTOTCORR",  # mm
            ]),
            "latitude": latitude,
            "longitude": longitude,
            "start": start_date.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"),
            "community": "AG",
            "format": "JSON",
        }
        logger.debug("Starting retrieval from NASA Power")
        req = requests.get(server, params=payload)
        if req.status_code != 200:
            raise RuntimeError(
                f"Failed retrieving POWER data, server returned HTTP "
                f"code: {req.status_code} on following URL {req.url}"
            )
        logger.debug("Successfully retrieved data from NASA Power")
        return req.json()

    def convert_NASAPower(cls, src: str,
                          fname: Optional[str] = None) -> str:
        r"""Convert a NASA Power weather file into this file format.

        Args:
            src: JSON file containing NASA Power data.
            fname: File name where the weather data should be written to.

        Returns:
            str: The generated file path.

        """
        if fname is None:
            fname = os.path.splitext(src)[0] + cls._default_ext
        if os.path.isfile(fname):
            return fname
        instance = cls.from_power(src, fname=fname)
        instance.write()
        return instance.fname

    @classmethod
    def from_location(cls, latitude: float, longitude: float,
                      fname: Optional[str] = None,
                      **kwargs) -> "BaseWeatherFile":
        r"""Create a weather file from a location by requesting NASA
        power weather data.

        Args:
            latitude: Location latitude (degrees).
            longitude: Location longitude (degrees).
            fname: File name where the weather data should be written to.
            **kwargs: Additional keyword arguments are passed to
                get_NASAPower_data.

        """
        contents = cls._from_power(
            cls.get_NASAPower_data(latitude, longitude, **kwargs))
        return cls(fname, contents=contents)

    @classmethod
    def from_power(cls, src: str,
                   fname: Optional[str] = None) -> "BaseWeatherFile":
        r"""Create a weather file from NASA power weather data.

        Args:
            src: JSON file containing NASA Power data.
            fname: File name where the weather data should be written to.

        """
        if fname is None:
            fname = os.path.splitext(src)[0] + cls._default_ext
        if os.path.isfile(fname):
            return cls(fname)
        with open(src, "r") as fd:
            contents = cls._from_power(json.load(fd))
        return cls(fname, contents=contents)

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


class CropModelEngine(BaseModelEngine):
    r"""Class for managining communication with a crop simulation model.

    Args:
        model_file: Path to one or more model input files.
        crop_name: Name of the crop.
        crop_variety: Name of the crop variety/cultivar.
        sow_date: Date that the crop should be sown.
        harvest_date: Date that the crop should be harvested.
        latitude: Field latitude to use to get weather data.
        longitude: Field longitude to use to get weather data.
        weather_file: Path to a file containing NASA power weather data.
        nasa_power_cache_dir: Directory where NASA Power weather files
            should be cached.
        **kwargs: Additional keywords arguments are passed along to
            BaseModelEngine.__init__.

    """

    EXPLICIT_PARAM = BaseModelEngine.EXPLICIT_PARAM + [
        "crop_name", "crop_variety",
        "sow_date", "harvest_date",
        "latitude", "longitude",
        "weather_file",
    ]
    WEATHER_FILE_TYPE = None

    def __init__(
            self,
            model_file: Optional[Union[str, List[str]]] = None,
            crop_name: Optional[str] = None,
            crop_variety: Optional[str] = None,
            sow_date: Optional[datetime.date] = None,
            harvest_date: Optional[datetime.date] = None,
            latitude: Optional[float] = None,
            longitude: Optional[float] = None,
            weather_file: Optional[str] = None,
            nasa_power_cache_dir: Optional[str] = None,
            **kwargs
    ):
        self.crop_name = crop_name
        self.crop_variety = crop_variety
        self.sow_date = sow_date
        self.harvest_date = harvest_date
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
            if not self.weather_file:
                self.weather_file = BaseWeatherFile.get_NASAPower_file(
                    self.latitude, self.longitude,
                    self.start_time.date(), self.end_time.date(),
                    cache_dir=self.nasa_power_cache_dir,
                )
            # TODO: Soil file?
        if self.sow_date is not None:
            self.action_map.pop("sow", None)
        if self.harvest_date is not None:
            self.action_map.pop("harvest", None)
        super().update_model_file()

    @property
    def location(self):
        r"""str: Description of the field location."""
        return self.model.location

    @property
    def field_area(self):
        r"""float: Field area"""
        return self.model.field_area

    @property
    def season_length(self):
        r"""int: Number of days in the growing season."""
        # TODO: Read this or set an average default?
        sow_date = (self.sow_date or self.start_time)
        harvest_date = (self.harvest_date or self.end_time)
        return (harvest_date - sow_date).days


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
