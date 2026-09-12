# Green-function examples

Each script uses one buried source and a regular receiver grid on the `Z = 0`
XY surface. Results are rendered with filled contours and written to
`examples/output/`; add `--show` to also open them interactively.

Install the package and plotting dependencies from the project root:

```bash
python -m pip install -e ".[examples]"
```

Run the interfaces independently:

```bash
python examples/regularized_dislocation.py
python examples/kelvin_stress.py
python examples/kelvin_stress_potential.py
python examples/laplace.py
python examples/dislocation_stress.py
python examples/dislocation_displacement.py
python examples/block_aca.py
python examples/volume_stress.py
```

The default receiver surface contains `21 x 21` points. For a quicker smoke
test, pass `--grid-size 7`; for smoother contours, pass a larger value.
