import os
from beam import Image, asgi
from typing import Optional
_apsimx_dir = "/workspace/ApsimX"
_n8n_dir = f"{_apsimx_dir}/n8n_tool"
_gym_dir = f"{_apsimx_dir}/ApsimXGym"
_python_version = "3.11"  # This value is hard coded in Dockerfile.beam_runner


class PatchedImage(Image):

    @classmethod
    def from_dockerfile(cls, path: str,
                        python_version: Optional[str] = None, **kwargs):
        image = super().from_dockerfile(path, **kwargs)
        if not python_version:
            pyver = _python_version
            python_version = f"python{_python_version}"
        elif python_version.startswith("micromamba"):
            pyver = python_version.split("micromamba", 1)[-1]
        else:
            pyver = python_version.split("python", 1)[-1]
        if pyver != _python_version:
            prefix = "ARG PYTHON_VERSION="
            assert (prefix + _python_version) in image.dockerfile
            image.dockerfile.replace(
                prefix + _python_version,
                prefix + pyver
            )
        image.python_version = python_version
        return image


image = (
    # A docker file must be used if both micromamba is needed and
    # external files will be used because beam.cloud uses a default
    # Dockerfile.runner and does not support COPY operations
    PatchedImage.from_dockerfile(
        os.path.join(os.path.dirname(__file__), "Dockerfile.beam_runner"),
        context_dir='.',
    )
    .micromamba()
    .with_envs({
        'PYTHONUNBUFFERED': '1',
        'APSIMX_DIR': _apsimx_dir,
    })
    .add_commands(commands=[
        f"micromamba install -n beta9 -f {_n8n_dir}/environment.yml",
        (
            "micromamba run -n beta9 dotnet build "
            f"{_apsimx_dir}/APSIM.Server/ZMQ+msgpack/APSIM.ZMQServer.csproj"
        ),
        f"micromamba run -n beta9 pip install -e {_gym_dir}",
    ])
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
