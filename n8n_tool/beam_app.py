import os
from beam import Image, asgi
_n8n_dir = os.path.abspath(os.path.dirname(__file__))


image = Image().from_dockerfile(
    os.path.join(_n8n_dir, "Dockerfile"),
    context_dir=os.path.dirname(_n8n_dir),
)


@asgi(
    name="apsimx-model",
    image=image,
    cpu=1.0,  # TODO: More for more models
    # memory=2048,
    # volumns:  # TODO: Populate a volumn with weather/soil data
)
def web_server(context):
    from main import app
    return app
