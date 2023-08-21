# CorgiPath: Customizable Path Planner

![Python version](https://img.shields.io/badge/python-3.8+-blue)
[![License](https://img.shields.io/github/license/ryul1206/corgipath)](https://github.com/ryul1206/corgipath/blob/main/LICENSE)

Here is a simple flow chart:

```mermaid
flowchart LR

c("Collision Layer")
s("Search-space Layer")

plan("Planning Module")
post("Postprocessing Module\n(optional)")
v("Matplot Viewer\n(optional)")

c --> plan
s --> plan

plan -.-> post
plan -.-> v
post -.-> v

style c stroke:#f66,stroke-width:2px
style s stroke:#f66,stroke-width:2px
style plan stroke:#f66,stroke-width:2px
style post stroke-width:2px,stroke-dasharray: 5 5
style v stroke-width:2px,stroke-dasharray: 5 5
```

## Installation

<!-- ```sh
poetry build
``` -->

## Build your own planner

Development in vscode

```
poetry config virtualenvs.in-project true
poetry config virtualenvs.path "./.venv"
```

## Contribute

Thanks for taking the time to contribute!

- We recommend developing in [a virtual environment using Poetry](https://python-poetry.org/docs/basic-usage#using-your-virtual-environment).
