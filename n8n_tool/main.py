import os
import re
import datetime
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import FastAPI, HTTPException
import logging


logging.basicConfig(level=logging.INFO)
interactive_models = {}


class Settings(BaseSettings):
    apsimx_dir: str = os.path.abspath(os.path.dirname(__file__))
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()


# Request payload for API, declared with Pydantic
class ModelInput(BaseModel):
    crop_name: str
    crop_variety: str
    latitude: float
    longitude: float
    year: int
    # TODO: ISO datestrings, but can these be datetime.datetime?
    start_date: str
    end_date: str
    # sow_date: str
    # harvest_date: str
    output_vars: list

    def run_apsimngpy(self):
        # TODO: Handle additional parameters:
        #   crop_variety, sow_date, harvest_date, output_vars
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
        from apsimx_gym.engine import ApsimXEngine
        kwargs.setdefault("actions", ["irrigate"])
        model = ApsimXEngine(
            crop_name=self.crop_name,
            crop_variety=self.crop_variety,
            latitude=self.latitude,
            longitude=self.longitude,
            year=self.year,
            start_time=datetime.datetime.fromisoformat(self.start_date),
            end_time=datetime.datetime.fromisoformat(self.end_date),
            # sow_date=datetime.datetime.fromisoformat(self.sow_date),
            # harvest_date=datetime.datetime.fromisoformat(self.harvest_date),
            model_dir=settings.apsimx_dir,
            **kwargs
        )
        model.start()
        return model

    def run_apsim_engine(self, **kwargs):
        model = self.start_apsim_engine(**kwargs)
        try:
            data = model.getvars(self.output_vars)
            model.resume()
            data = {k: [v] for k, v in data.items()}
            while model.is_running and not model.is_complete:
                idata = model.getvars(self.output_vars)
                model.resume()
                for k, v in idata.items():
                    data[k].append(v)
        finally:
            model.stop()
        return data


class InteractiveModelInput(ModelInput):
    action_step: int  # days
    actions: list

    def start_apsim(self, **kwargs):
        # TODO: Pass action_step
        import uuid
        idstr = str(uuid.uuid4())
        model = self.start_apsim_engine(actions=self.actions, **kwargs)
        interactive_models[idstr] = model
        return idstr


class InteractiveModelInputBase(BaseModel):

    @classmethod
    def check_model(cls, idstr, allow_stopped=False):
        if idstr not in interactive_models:
            raise HTTPException(
                status_code=404,
                detail=f"No model with id \"{idstr}\"",
            )
        model = interactive_models[idstr]
        if not (allow_stopped or model.is_running):
            raise HTTPException(
                status_code=404,
                detail=f"Model \"{idstr}\" is no longer running")
        return model

    def _do(self, model):
        raise NotImplementedError

    def do(self, idstr):
        model = self.check_model(idstr)
        return self._do(model)


class ModelSetInput(InteractiveModelInputBase):

    values: dict

    def _do(self, model):
        model.setvars(self.values)
        return {"status": "success"}


class ModelGetInput(InteractiveModelInputBase):

    names: list

    def _do(self, model):
        return model.getvars(self.names)


class ModelActionInput(InteractiveModelInputBase):

    action_name: str
    action_param: dict

    def _do(self, model):
        model.act(self.action_name, **self.action_param)
        return {"status": "success"}


app = FastAPI()


# Others async?
@app.post("/start")
async def start_model(input: ModelInput):
    return input.run_apsim_engine()
    # return input.run_apsimngpy()


@app.post("/start-interactive")
def start_interactive_model(input: InteractiveModelInput):
    idstr = input.start_apsim()
    return idstr


@app.post("/stop-interactive")
def stop_interactive_models():
    for k, v in interactive_models.items():
        if v.is_running:
            v.stop()
    interactive_models.clear()


@app.post("/prune-interactive")
def prune_interactive_models():
    stopped = [k for k, v in interactive_models.items()
               if not v.is_running]
    for k in stopped:
        del interactive_models[k]


@app.get("/interactive-model/{idstr}/status")
def interactive_model_status(idstr: str):
    model = InteractiveModelInputBase.check_model(idstr, allow_stopped=True)
    if not model.is_running:
        return {"status": "stopped"}
    return {"status": "running",
            "time": model.current_time}


@app.put("/interactive-model/{idstr}")
def interactive_model_set(idstr: str, input: ModelSetInput):
    return input.do(idstr)


@app.get("/interactive-model/{idstr}")
def interactive_model_get(idstr: str, input: ModelGetInput):
    return input.do(idstr)


@app.post("/interactive-model/{idstr}/act")
def interactive_model_act(idstr: str, input: ModelActionInput):
    return input.do(idstr)


@app.post("/interactive-model/{idstr}/continue")
def interactive_model_continue(idstr: str):
    model = InteractiveModelInputBase.check_model(idstr)
    model.resume(wait=True)
    return {"status": "success"}


@app.post("/interactive-model/{idstr}/complete")
def interactive_model_complete(idstr: str):
    model = InteractiveModelInputBase.check_model(idstr)
    model.fast_forward()
    return {"status": "success"}


@app.post("/interactive-model/{idstr}/restart")
def interactive_model_restart(idstr: str):
    model = InteractiveModelInputBase.check_model(idstr)
    model.rewind()
    return {"status": "success"}


@app.post("/interactive-model/{idstr}/stop")
def interactive_model_stop(idstr: str):
    model = InteractiveModelInputBase.check_model(idstr)
    model.stop()
    return {"status": "success"}


@app.post("/interactive-model/{idstr}/scrub")
def interactive_model_scrub(idstr: str, time: int | str):
    model = InteractiveModelInputBase.check_model(idstr)
    if isinstance(time, str) and re.fullmatch(r'[-+]?[1-9][0-9]*', time):
        time = int(time)
    model.scrub(time)
    return {"status": "success"}
