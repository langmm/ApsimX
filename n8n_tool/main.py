import re
from fastapi import FastAPI
import logging
from config import (
    Settings, InteractiveModelRegistry,
    ModelInput, ModelSetInput, ModelGetInput,
    ModelActionInput, InteractiveModelInput
)


logging.basicConfig(level=logging.INFO)


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


# @app.post("/start-apsimngpy")
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
        return model._model.getvars(input.state_variables)


@app.post("/interactive-model/{idstr}/act")
async def interactive_model_act(idstr: str, input: ModelActionInput):
    async with interactive_models.valid_model(idstr) as model:
        model._model.act(input.action, **input.parameters)
        return {"status": "success"}


@app.get("/interactive-model/{idstr}/trace")
async def interactive_model_trace(idstr: str):
    async with interactive_models.valid_model(idstr,
                                              allow_stopped=True) as model:
        return model._trace


# Disabled until report generation can be debugged
# @app.get("/interactive-model/{idstr}/results")
# async def interactive_model_results(idstr: str):
#     async with interactive_models.valid_model(idstr,
#                                               allow_stopped=True) as model:
#         return model.get_results()


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
