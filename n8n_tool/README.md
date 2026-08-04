# ApsimX n8n tools

This directory includes the necessary components for deploying a REST API for running ApsimX models and creating n8n tools from that REST API.

## Files

- config.py - Pydantic models for REST API requests that call ApsimX via apsimx_gym
- main.py - fastapi application
- make_n8n_form.py - Utilities for creating/updating ApsimX n8n tools

## REST API

To launch the server run the following from the n8n_tool directory

```
fastapi run --host 0.0.0.0 --port 5000 main.py
```

To run the tests

```
pytest -svx n8n_tool/tests/test_main.py
```

## Docker

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

## Beam Deployment

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

To run the tests using the deployment on beam

```
export APSIMX_REMOTE_ADDRESS=<BEAM_URL>
pytest -svx n8n_tool/tests/test_main.py
```

## n8n Tool

The n8n_tool/make_n8n_form.py Python script provides CLI tools for managing an n8n tool that uses the beam service.

To access the n8n api, a valid API key must be passed via the ``X_N8N_API_KEY`` environment variable. The address of the service on beam can be provided via the ``APSIMX_REMOTE_ADDRESS`` environment variable or via the ``--publish-for-address`` CLI argument.

```
export X_N8N_API_KEY=<N8N_CREDENTIALS>
export APSIMX_REMOTE_ADDRESS=<BEAM_URL>
```

To create an n8n tool that uses the apsimx model service

```
python n8n_tool/make_n8n_form.py create --name start
```

To update an existing n8n tool that uses the apsimx model service

```
python n8n_tool/make_n8n_form.py update --name start
```

## Development

Procedure on update to apsimx_gym or n8n_tool directory:

1. Run local n8n_tool/tests
1. Build docker & run docker tests (ensure container shutdown to prevent interference with future local tests)
1. Redeploy to beam
1. Check that docs render correctly
1. Update APSIMX_REMOTE_ADDRESS in test environment
1. Run remote n8n_tool/tests
1. Update the n8n tool

## To do

### Short term

- Add bearer token authentication when using pod deployment & redeploy since the pod deployment does not seem to implement beam bearer credential
- Add support for providing a weather/soil file via upload

### Long term

- Improved documentation
- Move apsimx_gym into pydantic models to allow autogeneration of n8n forms
- Test build with different version of Python?
- Endpoint for getting a list of state variables (this would require parsing C# files)
- Auto test API examples/defaults
- Add option to upload a crop model file?
