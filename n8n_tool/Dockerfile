FROM condaforge/miniforge3

ARG sourcedir
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# Copy local repository
COPY --exclude=./n8n_tool/main.py --exclude=./n8n_tool/config.py --exclude=./n8n_tool/launch_local.sh --exclude=./ApsimXGym . /ApsimX/
WORKDIR /ApsimX

# Create environment
RUN conda env create -f n8n_tool/environment.yml -n apsimx

# Install ApsimX
RUN conda run -n apsimx dotnet build APSIM.Server/ZMQ+msgpack/APSIM.ZMQServer.csproj
# RUN conda run -n apsimx dotnet build Models/Models.csproj
# RUN conda run -n apsimx dotnet build APSIM.Cli/APSIM.Cli.csproj

# Install apsimNGpy
# RUN conda run -n apsimx pip install apsimNGpy
# RUN conda run -n apsimx apsim_bin_path --update "/ApsimX/bin"
# RUN conda run -n apsimx apsim_bin_path --update "/ApsimX/bin/Debug/net8.0"

# Install ApsimXGym
COPY ./ApsimXGym/ /ApsimX/ApsimXGym/
WORKDIR /ApsimX/ApsimXGym
RUN conda run -n apsimx pip install -e .

# Launch
WORKDIR /home
COPY ./n8n_tool/main.py .
COPY ./n8n_tool/config.py .
COPY ./n8n_tool/launch_local.sh .

ARG APP_PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PORT=${APP_PORT}
ENV APSIMX_DIR=/ApsimX
EXPOSE ${APP_PORT}

WORKDIR /home
CMD ["conda", "run", "--no-capture-output", "-n", "apsimx", "./launch_local.sh"]

