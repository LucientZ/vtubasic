# VTubasic

Very very simple and basic VTuber runtime. Do not use this if your goal is efficiency!

## Installation

This program requires some pip packages to run. As such, it's recommended to create a virtual environment with the following command:

```bash
python -m venv venv
```

To install the dependencies once the environment is activated:

```bash
pip install -r requirements.txt
```

Finally, to run the program:

```bash
python main.py
```

## Controls

### Runtime

- `w` Toggle drawing mesh
- `r` Reset physics
- `z` Set background to red
- `x` Set background to blue
- `c` Set background to green
- `(0-9)` Expressions

### Editor (broken)

- `[` Move back a layer
- `]` Move forward a layer
- `click` Add vertex
- `z` Remove the most recent vertex
    - This includes any existing vertices
- `r` Reset vertices to last loaded mesh
- `ctrl-s` Overwrites the loaded mesh with current mesh


## How it works

All the documentation can be found [with this link](https://lucientz.github.io/vtubasic/).