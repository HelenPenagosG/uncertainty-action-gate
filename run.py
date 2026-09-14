"""run.py — Punto de entrada del runner de experimentos.

Envuelve `rhlab.runner.main` para poder ejecutar desde la raíz del repo:

    py -3.12 run.py --condition C --runs 3 --mock

(equivalente a `py -3.12 -m rhlab.runner ...`).
"""
from rhlab.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
