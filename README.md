# MixingCut Solver
MixingCut is a solver for the MaxCut SDP relaxation via mixing cut algorithm proposed by Po-Wei Wang et al [1]. This 
can solve general large scale instances of the MaxCut SDP relaxation problem.

$$
\begin{align*}
\min_{X} &\text{tr}(QX) \\
s.t. &X_{ii} = 1\\
&X \succeq 0
\end{align*}
$$

# Usage

The solver can be used via the command line with the following interface:

```bash
./mixingcut --input-path <input_file> --output-path <output_file> -rank <rank> --tolerance <tolerance> --max-iters <max_iter> --index-correction <index-correction> --step-rule <step_rule> --dual-bound <dual_bound> --verbose <verbose> 
```

Where:
- `<input_file>` is the path to the file containing the input data for the matrix ``Q``. 
- `<output_file>` is the path to the file where the output will be written. Default is ``output.txt``.
- `<rank>` is the rank of the low rank approximation, and ``rank = 0`` selects ``2log2(n)``, and ``rank = 1`` 
  selects ``sqrt(2n)`` as the rank. Default is ``0`` 
- `<tolerance>` is the tolerance for the stopping criterion between iterations. If the objective improvement is less 
  than this number, solver termination occures. Default is ``1e-2``. 
- `<max_iter>` is the maximum number of iterations. Default is ``1000``.
- `<step_rule>` is the update rule for the mixing cut algorithm. Default is ``coord_no_step``. Options are ``grad``, 
  ``grad_adv``, ``coord`` and ``coord_no_step``. 
- `<index_correction>` is a for reading the input graph file, if the vertices are 0-indexed or 1-indexed. Default is 
  ``1``. 
- `<dual_bound>` is the flag for computing the dual bound for the SDP relaxation. Default is ``0``. A value of ``1`` 
  will compute the dual bound (this is an expensive(non computable) operation for large(huge) instances).  
- `<verbose>` is the flag for printing the output to the console. Default is ``1``. To disable output set to ``0``.

However, only the input file is required. The solver will use the default values for the other parameters if not 
specified in the command line with the following command: 

```bash
./mixingcut --input-path <input_file>
```

# Expected File Format
The solver reads the input from a file in the following format: Where `n` is the number of vertices, `i` and `j` are 
the vertices of the edge and `q_ij` is the weight of the edge. In the case where ``q_ij`` is not specified, the 
default value is 1.

```
n
i_1 j_1 q_{i_1,j_1}
i_2 j_2 q_{i_2,j_2}
...
i_m j_m q_{i_m,j_m}
```

# Output File Format
The solver writes a rounded solution to a file in the following format: Where ``obj_sdp`` is the value of the 
SDP relaxation,``obj_rounded_sol`` is the value of the rounded solution and `x_i` is the value of the vertex `i`. 

```
obj_sdp
obj_rounded_sol
x_0
x_1
...
x_{n-1}
```

# More Info 

The MixingCut Solver is based on the ideas proposed by the following paper:

```bibtex
@article{wang2017mixing,
	title = {The Mixing method: coordinate descent for low-rank semidefinite programming},
	author = {Po-Wei Wang and Wei-Cheng Chang and J. Zico Kolter},
	journal = {arXiv preprint arXiv:1706.00476},
	year = {2017}
}
```
