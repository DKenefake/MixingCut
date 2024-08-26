# MixingCut Solver
Solver for the MaxCut SDP relaxation via mixing cut algorithm proposed by Po-Wei Wang et al [1]. Where theis SDP is solved via a low rank reformulation that is exact under certain rank conditions.

$$
\begin{align*}
\min_{X} &\text{tr}(QX) \\
s.t. &X_{ii} = 1\\
&X \succeq 0
\end{align*}
$$

# Usage

The solver can be used as follows:

```bash
mixing_cut_solver -i <input_file> -o <output_file> -r <rank> -t <tolerance> -m <max_iter>
```

Where:
- `<input_file>` is the path to the file containing the input data.
- `<output_file>` is the path to the file where the output will be written.
- `<rank>` is the rank of the low rank approximation.
- `<tolerance>` is the tolerance for the stopping criterion.
- `<max_iter>` is the maximum number of iterations.

# Expected File Format
The solver reads the input from a file in the following format: Where `n` is the number of vertices, `i` and `j` are the vertices of the edge and `q_ij` is the weight of the edge.

```
n 
i j q_ij
....
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
