import os
import re
import uuid
import datetime
import asyncio
import contextlib
from pydantic import BaseModel, PrivateAttr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import FastAPI, HTTPException
import logging


logging.basicConfig(level=logging.INFO)


class Settings(BaseSettings):
    apsimx_dir: str = os.path.abspath(os.path.dirname(__file__))
    model_config = SettingsConfigDict(env_file=".env")


class InteractiveModelRegistry:

    def __init__(self):
        self._models = {}
        self._lock = asyncio.Lock()
        self._in_use = []

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
    crop_name: str
    crop_variety: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    year: int | None = None
    start_time: str | datetime.datetime | None = None
    end_time: str | datetime.datetime | None = None
    # sow_date: str | datetime.datetime | None = None
    # harvest_date: str | datetime.datetime | None = None
    timestep: int | datetime.timedelta | None = None
    trace_vars: list = ["[Wheat].Grain.Total.Wt"]
    _model: PrivateAttr(default=None)
    _trace: PrivateAttr(default=None)

    @field_validator('start_time', 'end_time')  # 'sow_date', 'harvest_date')
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

    def model_post_init(self, context):
        self._model = None
        self._trace = None
        return super().model_post_init(context)

    def log_and_continue(self, wait: bool = False):
        if not (self._model and self._model.is_running
                and not self._model.is_complete):
            return
        idata = None
        if self.trace_vars:
            idata = self._model.getvars(self.trace_vars)
        if self.timestep is None:
            self._model.resume(wait=wait)
        else:
            self._model.fast_forward(self.timestep)
        if self.trace_vars:
            if self._trace is None:
                self._trace = {k: [v] for k, v in idata.items()}
            else:
                for k, v in idata.items():
                    self._trace[k].append(v)

    def run_apsimngpy(self):
        # TODO:
        #   - Debug runtime where there is an error in pythonnet
        #   - Handle additional parameters:
        #       crop_variety, sow_date, harvest_date, trace_vars
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
            exclude=["timestep", "trace_vars", "timeout"],
        ))
        kwargs.setdefault("actions", ["irrigate"])
        self._model = ApsimXEngine(**kwargs)
        self._model.start()

    def run_apsim_engine(self, **kwargs):
        self.start_apsim_engine(**kwargs)
        try:
            while self._model.is_running and not self._model.is_complete:
                self.log_and_continue()
        finally:
            self._model.stop()
        return self._trace


class InteractiveModelInput(ModelInput):
    # TODO: Return trace/results?
    trace_vars: list | None = None
    actions: list = ["irrigate"]
    timeout: int = 300
    _idstr: PrivateAttr(default=None)
    _model_lock: PrivateAttr(default=None)
    _model_accessed: PrivateAttr(default=None)
    _shutdown_after_wait: PrivateAttr(default=None)

    @field_validator('timeout')
    @classmethod
    def check_timeout(cls, v):
        if v < 0 or v > 300:
            return 300
        return v

    def start_apsim_engine(self, **kwargs):
        super().start_apsim_engine(**kwargs)
        self._idstr = str(uuid.uuid4())
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
                async with asyncio.timeout(self.timeout):
                    await self._model_accessed.wait()
                self._model_accessed.clear()
            except TimeoutError:
                if is_running:
                    async with self._model_lock:
                        if self._model.is_running:
                            self._model.stop()
                        assert not self._model.is_running
                break


class ModelSetInput(BaseModel):

    values: dict


class ModelGetInput(BaseModel):

    names: list


class ModelActionInput(BaseModel):

    action_name: str
    action_param: dict


app = FastAPI()
settings = Settings()
interactive_models = InteractiveModelRegistry()


# Others async?
@app.get("/")
async def status():
    return (
        f"This is an ApsimX server with {interactive_models.size()} "
        f"interactive models currently running"
    )


@app.post("/start")
async def start_model(input: ModelInput):
    return input.run_apsim_engine(model_dir=settings.apsimx_dir)


# @app.post("/start")
# async def start_apsimngpy_model(input: ModelInput):
#     return input.run_apsimngpy()


@app.post("/start-interactive")
async def start_interactive_model(input: InteractiveModelInput):
    input.start_apsim_engine(model_dir=settings.apsimx_dir)
    await interactive_models.add(input)
    return input._idstr


@app.post("/stop-interactive")
async def stop_interactive_models():
    await interactive_models.clear()


@app.post("/prune-interactive")
async def prune_interactive_models():
    await interactive_models.clear(dont_stop=True)


@app.get("/interactive-model/{idstr}/status")
async def interactive_model_status(idstr: str):
    async with interactive_models.valid_model(idstr,
                                              allow_stopped=True) as model:
        if not model._model.is_running:
            return {"status": "stopped"}
        return {"status": "running",
                "time": model._model.current_time}


@app.put("/interactive-model/{idstr}")
async def interactive_model_set(idstr: str, input: ModelSetInput):
    async with interactive_models.valid_model(idstr) as model:
        model._model.setvars(input.values)
        return {"status": "success"}


@app.get("/interactive-model/{idstr}")
async def interactive_model_get(idstr: str, input: ModelGetInput):
    async with interactive_models.valid_model(idstr) as model:
        return model._model.getvars(input.names)


@app.post("/interactive-model/{idstr}/act")
async def interactive_model_act(idstr: str, input: ModelActionInput):
    async with interactive_models.valid_model(idstr) as model:
        model._model.act(input.action_name, **input.action_param)
        return {"status": "success"}


@app.post("/interactive-model/{idstr}/continue")
async def interactive_model_continue(idstr: str):
    async with interactive_models.valid_model(idstr) as model:
        model.log_and_continue(wait=True)
        return {"status": "success"}


@app.post("/interactive-model/{idstr}/complete")
async def interactive_model_complete(idstr: str):
    async with interactive_models.valid_model(idstr) as model:
        model._model.fast_forward()
        return {"status": "success"}


@app.post("/interactive-model/{idstr}/restart")
async def interactive_model_restart(idstr: str):
    async with interactive_models.valid_model(idstr) as model:
        model._model.rewind()
        return {"status": "success"}


@app.post("/interactive-model/{idstr}/stop")
async def interactive_model_stop(idstr: str):
    async with interactive_models.valid_model(idstr) as model:
        model._model.stop()
        return {"status": "success"}


@app.post("/interactive-model/{idstr}/scrub")
async def interactive_model_scrub(idstr: str, time: int | str):
    async with interactive_models.valid_model(idstr) as model:
        if isinstance(time, str) and re.fullmatch(r'[-+]?[1-9][0-9]*', time):
            time = int(time)
        model._model.scrub(time)
        return {"status": "success"}
