import os
import json
import uuid
import datetime
import asyncio
import contextlib
import sqlite3
import pandas as pd
from typing import List
from pydantic import BaseModel, PrivateAttr, field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import HTTPException
from apsimx_gym.engine import ApsimXEngine
_actions = list(ApsimXEngine.AVAILABLE_ACTION_MAP.keys())


class Settings(BaseSettings):
    apsimx_dir: str = os.path.abspath(os.path.dirname(__file__))
    model_config = SettingsConfigDict(env_file=".env")


class InteractiveModelRegistry:

    def __init__(self):
        self._models = {}
        self._lock = asyncio.Lock()
        self._in_use = []

    def __del__(self):
        for k in list(self._models.keys()):
            asyncio.run(self._safe_remove(k))

    @contextlib.asynccontextmanager
    async def valid_model(self, idstr, allow_stopped=False):
        model = None
        async with self._lock:
            if idstr not in self._models:
                raise HTTPException(
                    status_code=404,
                    detail=f"No model with id \"{idstr}\"",
                )
            self._in_use.append(idstr)
            model = self._models[idstr]
        async with model._model_lock:
            try:
                if not (allow_stopped or model._model.is_running):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model \"{idstr}\" is no longer running")
                yield model
            finally:
                async with self._lock:
                    self._in_use.remove(idstr)

    async def add(self, model):
        async with self._lock:
            assert model._idstr not in self._models
            self._models[model._idstr] = model

    async def _safe_remove(self, idstr, dont_stop=False):
        if idstr not in self._models:
            return True
        if idstr in self._in_use:
            return False
        model = self._models[idstr]
        async with model._model_lock:
            if model._model.is_running and not dont_stop:
                model._model.stop()
            if not (dont_stop and model._model.is_running):
                model.finalize_model(skip_results=True)
                del self._models[idstr]
        return True

    async def remove(self, idstr, **kwargs):
        while True:
            async with self._lock:
                if await self._safe_remove(idstr, **kwargs):
                    return

    async def clear(self, ids=None, **kwargs):
        while True:
            async with self._lock:
                if ids is None:
                    ids = list(self._models.keys())
                if not any(idstr in self._models for idstr in ids):
                    return
                ids = [
                    idstr for idstr in ids
                    if not await self._safe_remove(idstr, **kwargs)
                ]

    async def size(self):
        async with self._lock:
            return len(self._models)


# Request payload for API, declared with Pydantic
class ModelInput(BaseModel):
    crop_name: str = Field(description="that will be simulated")
    crop_variety: str | None = Field(
        None, description="that will be simulated")
    latitude: float | None = Field(
        None, description="used to get weather data (degrees)")
    longitude: float | None = Field(
        None, description="used to get weather data (degrees)")
    year: int | None = Field(
        None, description=(
            "used to get weather data. Overrides the year in the "
            "default/provided start_time, end_time, sow_date, and/or "
            "harvest_date"
        ))
    start_time: datetime.datetime | None = Field(
        None, description="of simulation (ISO 8601 format)")
    end_time: datetime.datetime | None = Field(
        None, description="of simulation (ISO 8601 format)")
    sow_date: datetime.datetime | None = Field(
        None, description="for the simulated crop (ISO 8601 format)"
    )
    harvest_date: datetime.datetime | None = Field(
        None, description="for the simulated crop (ISO 8601 format)"
    )
    timestep: int | datetime.timedelta | None = Field(
        None, description=(
            "between records of state variables (days)"
        ))
    state_variables: List[str] = Field(
        ["[Clock].Today", "[Wheat].Grain.Total.Wt"], description=(
            "that should be recorded at each reported time step "
            "(comma separated list)"
        ))
    _idstr: PrivateAttr(default=None)
    _model: PrivateAttr(default=None)
    _trace: PrivateAttr(default=None)
    _results: PrivateAttr(default=None)

    @field_validator('start_time', 'end_time', 'sow_date', 'harvest_date',
                     mode="before")
    @classmethod
    def check_datetime(cls, v):
        if isinstance(v, str):
            return datetime.datetime.fromisoformat(v)
        return v

    @field_validator('timestep')
    @classmethod
    def check_timedelta(cls, v):
        if isinstance(v, int):
            if v <= 0:
                return None
            return datetime.timedelta(days=v)
        return v

    @field_validator('state_variables', mode="before")
    @classmethod
    def check_list(cls, v):
        if isinstance(v, str):
            return [vv.strip() for vv in v.split(",")]
        return v

    def model_post_init(self, context):
        self._model = None
        self._trace = None
        self._results = None
        return super().model_post_init(context)

    def store_results(self):
        if os.path.isfile(self._model.output_file):
            # This currently errors due to missing report
            conn = sqlite3.connect(self._model.output_file)
            df = pd.read_sql_query("SELECT * FROM Report", conn)
            self._results = df.to_json()

    def get_results(self):
        if self._model.is_running and not self._model.is_complete:
            raise HTTPException(
                status_code=404,
                detail=f"Model \"{self._idstr}\" is still running"
            )
        if not os.path.isfile(self._model.output_file):
            if self._results:
                return self._results
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Output file for model \"{self._idstr}\" does not "
                    f"exist"
                )
            )
        self.store_results()
        return self._results

    def finalize_model(self, skip_results: bool = False):
        self._model.cleanup(remove_output=True)

    def log_and_continue(self, wait: bool = False):
        if not (self._model and self._model.is_running
                and not self._model.is_complete):
            return
        idata = None
        if self.state_variables:
            idata = self._model.getvars(self.state_variables)
        if self.timestep is None:
            self._model.resume(wait=wait)
        else:
            self._model.fast_forward(self.timestep)
        if self.state_variables:
            if self._trace is None:
                self._trace = {k: [v] for k, v in idata.items()}
            else:
                for k, v in idata.items():
                    self._trace[k].append(v)

    def run_apsimngpy(self):
        # TODO:
        #   - Debug runtime where there is an error in pythonnet
        #   - Handle additional parameters:
        #       crop_variety, sow_date, harvest_date, state_variables
        from apsimNGpy.core.apsim import ApsimModel
        LONLAT = (self.latitude, self.longitude)
        with ApsimModel(self.crop_name) as model:
            model.get_weather_from_web(
                lonlat=LONLAT,
                start=max(self.year, 1990),
                end=min(self.year, 2001),
            )
            model.get_soil_from_web(
                simulations=None,
                lonlat=LONLAT,
                source="ssurgo",
            )
            model.run()
            return model.results.to_json()

    def start_apsim_engine(self, **kwargs):
        assert not self._model
        from apsimx_gym.engine import ApsimXEngine
        kwargs.update(self.model_dump(
            exclude_none=True,
            exclude=["timestep", "state_variables", "wait_time"],
        ))
        kwargs.setdefault("actions", ["irrigate"])
        self._idstr = str(uuid.uuid4())
        self._model = ApsimXEngine(**kwargs)
        self._model.start()

    def run_apsim_engine(self, **kwargs):
        self.start_apsim_engine(**kwargs)
        try:
            while self._model.is_running and not self._model.is_complete:
                self.log_and_continue()
        finally:
            self._model.stop(cleanup=False)
            self.finalize_model()
        return self._trace


class InteractiveModelInput(ModelInput):
    actions: List[str] = Field(
        _actions, description=(
            "to make available at each timestep "
        ), json_schema_extra={
            "items": {
                "type": "str",
                "enum": _actions,
            },
        }
    )
    wait_time: int = Field(
        300, description=(
            "that the model should be kept alive at each time "
            "step awaiting an interactive command (seconds)"
        ))
    _model_lock: PrivateAttr(default=None)
    _model_accessed: PrivateAttr(default=None)
    _shutdown_after_wait: PrivateAttr(default=None)

    @field_validator('state_variables', 'actions', mode="before")
    @classmethod
    def check_list(cls, v):
        if isinstance(v, str):
            return [vv.strip() for vv in v.split(",")]
        return v

    @field_validator('wait_time')
    @classmethod
    def check_wait_time(cls, v):
        if v < 0 or v > 300:
            return 300
        return v

    def start_apsim_engine(self, **kwargs):
        super().start_apsim_engine(**kwargs)
        self._model_lock = asyncio.Lock()
        self._model_accessed = asyncio.Event()
        self._shutdown_after_wait = asyncio.create_task(
            self.shutdown_after_wait())

    async def shutdown_after_wait(self):
        is_running = True
        while is_running:
            if self._model_lock is None:
                return
            async with self._model_lock:
                if self._model is None:
                    return
                is_running = self._model.is_running
            if not is_running:
                break
            try:
                async with asyncio.timeout(self.wait_time):
                    await self._model_accessed.wait()
                self._model_accessed.clear()
            except TimeoutError:
                if is_running:
                    async with self._model_lock:
                        if self._model.is_running:
                            self._model.stop()
                        self.finalize_model()
                        assert not self._model.is_running
                break


class ModelSetInput(BaseModel):

    values: dict = Field(
        description=(
            "mapping between state variable names and values they "
            "should be set to (json object)"
        ))

    @field_validator('values', mode="before")
    @classmethod
    def check_json(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


class ModelGetInput(BaseModel):

    state_variables: List[str] = Field(
        description="to get values for (comma separated list)"
    )

    @field_validator('state_variables', mode="before")
    @classmethod
    def check_list(cls, v):
        if isinstance(v, str):
            return [vv.strip() for vv in v.split(",")]
        return v


class ModelActionInput(BaseModel):

    # TODO: Get values
    action: str = Field(
        description="to perform",
        json_schema_extra={"enum": _actions},
    )
    parameters: dict = Field(
        {},
        description="describing the management action (json object)"
    )

    @field_validator('parameters', mode="before")
    @classmethod
    def check_json(cls, v):
        if isinstance(v, str):
            if not v:
                return {}
            return json.loads(v)
        return v
