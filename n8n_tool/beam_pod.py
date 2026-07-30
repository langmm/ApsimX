import os
from beam import Image, Pod
_n8n_dir = os.path.abspath(os.path.dirname(__file__))


image = Image().from_dockerfile(
    os.path.join(_n8n_dir, "Dockerfile"),
    context_dir=os.path.dirname(_n8n_dir),
)

pod = Pod(
    name='apsimx-model',
    image=image,
    cpu=2,
    ports=[8000],
    # entrypoint=[
    #     "conda", "run", "--no-capture-output", "-n", "apsimx",
    #     "fastapi", "run", "main.py",
    # ],
)

# res = pod.create()
# print(f"Pod at {res.url}")
