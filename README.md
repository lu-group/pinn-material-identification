# Physics-informed neural networks for material identification

The data and code for the paper [W. Wu, M. Daneker, M. A. Jolley, K. T. Turner, & L. Lu. Effective data sampling strategies and boundary condition constraints of physics-informed neural networks for identifying material properties in solid mechanics. *arXiv preprint arXiv:2211.15423*, 2022](https://arxiv.org/abs/2211.15423).

## Code

All data and code are in the folder [src](src). The code depends on the deep learning package [DeepXDE](https://github.com/lululxvi/deepxde) v1.6.2. 

- Inverse one-dimensional PDE problems
    - Time-dependent longitudinal vibration
        - [Soft constraints](src/1D/time_dependent_longitudinal_vibration_inverse/soft_constraints.py)
    - Time-dependent lateral vibration
        - [Soft constraints](src/1D/time_dependent_lateral_vibration_inverse/soft_constraints.py)
        - [Hard constraints case 1](src/1D/time_dependent_lateral_vibration_inverse/hard_constraints_1.py)
        - [Hard constraints case 2](src/1D/time_dependent_lateral_vibration_inverse/hard_constraints_2.py)
- Inverse two-dimensional PDE problems
    - Time-independent with linear elastic material
        - [Soft constraints](src/2D/linear_elastic_steady_state_inverse/soft_constraints.py)
        - [Hard constraints with a discontinuous function](src/2D/linear_elastic_steady_state_inverse/hard_constraints_discontinous_func.py)
        - [Hard constraints with a smooth function](src/2D/linear_elastic_steady_state_inverse/hard_constraints_smooth_func.py)
    - Time-independent with hyperelastic material
        - [Hard constraints with a discontinuous function](src/2D/hyperelastic_steady_state_inverse/hard_constraints_discontinous_func.py)
    - Time-dependent with linear elastic material
        - [Hard constraints with a discontinuous function](src/2D/linear_elastic_dynamics_inverse/hard_constraints_discontinous_func.py)

## Cite this work

If you use this data or code for academic research, you are encouraged to cite the following paper:

```
@article{wu2022materialidentification,
  title={Effective data sampling strategies and boundary condition constraints of physics-informed neural networks for identifying material properties in solid mechanics}, 
  author={Wensi Wu and Mitchell Daneker and Matthew A. Jolley and Kevin T. Turner and Lu Lu},
  year={2022},
  eprint={2211.15423},
  archivePrefix={arXiv}
}
```

## Questions

To get help on how to use the data or code, simply open an issue in the GitHub "Issues" section.
