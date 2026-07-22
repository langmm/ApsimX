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
docker run -p 5000:80 -d apsimx
```

To select a different host/container port

```
docker build -f n8n_tool/Dockerfile --build-arg APP_PORT=${CONTAINER_PORT} -t apsimx .
docker run -p ${HOST_PORT}:${CONTAINER_PORT} -d apsimx
```