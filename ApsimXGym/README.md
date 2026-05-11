# ApsimXGym

ApsimXGym provides a [gymnasium](https://gymnasium.farama.org/) environment for reinforcement learning using [ApsimX](https://apsimnextgeneration.netlify.app/).

## Requirements

- .NET 8.0 SDK library
- Gtk3 and GtkSourceView

## Installation from source

After installing the requirments above, ApsimX can be installed from the ApsimX repository root directory via

```
dotnet build ApsimX.sln
```

See instructions [here](https://apsimnextgeneration.netlify.app/development/compile/) if you encounter errors when installing ApsimX.

Then the gym can be installed from the ApsimXGym directory via

```
pip install .
```

## Command line interface

To run the engine in a loop.

```
python -m apsimx_gym run /path/to/file/Wheat.apsimx
```