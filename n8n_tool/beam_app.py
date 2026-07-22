from beam import Image, asgi


image = Image().from_dockerfile(
    "./Dockerfile",
    context=".."
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
