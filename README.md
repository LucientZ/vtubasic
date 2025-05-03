# vtubasic

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

### Editor (broken)

- `w` Toggle drawing mesh
- `[` Move back a layer
- `]` Move forward a layer
- `click` Add vertex
    - This currently just prints these to the console


## How it works

An example model is provided for reference of how to configure a model. Copy its style for now while I eventually take the time to write a real guide.