To launch the server run the following from the n8n_tool directory

```
fastapi run --host 0.0.0.0 --port 5000 main.py
```

To run the server tests

```
pytest -svx n8n_tool/tests/test_main.py::test_interactive
```

To build the docker image this run the following from the ApsimX root directory

```
docker build -f n8n_tool/Dockerfile -t apsimx .
```

To run the server in the docker image

```
docker run -p 5000:8000 -d apsimx
```

To select a different host/container port

```
docker build -f n8n_tool/Dockerfile --build-arg APP_PORT=${CONTAINER_PORT} -t apsimx .
docker run -p ${HOST_PORT}:${CONTAINER_PORT} -d apsimx
```

To deploy pod based application to beam.cloud

```
beam deploy n8n_tool/beam_pod.py:pod
```

To serve docker/python based ASGI application. 

```
beam serve n8n_tool/beam_asgi.py:web_server
beam serve n8n_tool/beam_docker.py:web_server
```

The asgi version does not work currently and causes the following error

```
/micromamba/envs/beta9/bin/python3: Error while finding module specification for 'beta9.runner.serve' (ModuleNotFoundError: No module named 'beta9')
```

# TODO:
- Redeploy
- Move pydantic classes into separate file or adjust entry point
- Script to regenerate & submit n8n form from pydantic

- Test build with different version of Python?
