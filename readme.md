# SUMO: Subspace-Aware Moment-Orthogonalization

Official implementation of the paper **"SUMO: Subspace-Aware Moment-Orthogonalization for Accelerating Memory-Efficient LLM Training".**

SUMO is a next-generation optimizer that bridges the gap between memory-efficient low-rank training and high-performance geometric optimization. By performing **exact SVD-based orthogonalization** within a dynamically adapted low-dimensional subspace, SUMO accelerates convergence while requiring significantly less memory than previous state-of-the-art methods.

---

## 🚀 Key Features

* **Memory Efficiency:** Reduces memory requirements by up to 20% compared to state-of-the-art methods like GaLore by relying solely on the first-order moment.
* **Faster Convergence:** Achieves up to **~1.6x speedup** in convergence compared to GaLore and approximate orthogonalization methods.
* **Superior Stability:** Replaces unstable approximations like Newton-Schulz with **exact SVD** orthogonalization, mitigating errors in ill-conditioned landscapes.
* **Subspace-Aware:** Explicitly aligns optimization steps with the spectral characteristics of the loss landscape within a dynamically adapted low-dimensional subspace.

---

## 📊 Comparison with State-of-the-Art

| Feature | **SUMO**        | **Adam** | **Shampoo**  | **SOAP**        | **GaLore**      |
| :--- |:----------------|:---------|:-------------|:----------------|:----------------|
| **Optim. States Memory** | $nr+mr$         | $2mn$    | $m^2+n^2$    | $2mn+2m^2+2n^2$ | $2nr+mr$        |
| **Subspace-Aware** | ✅               | ❌        | ❌            | ❌               | ✅               |
| **Orthogonalization** | **Exact (SVD)** | ❌        | Approx       | Approx          | ❌               |
| **Comput. Complexity** | $O(mnr+mn^2/K)$ | $O(mn)$  | $O(m^3+n^3)$ | $O(m^3+n^3)$    | $O(mnr+mn^2/K)$ |



---

## 🧠 How it Works

SUMO optimizes training through four primary blocks:

1.  **Adaptive Subspace Selection:** Leverages Randomized-SVD to efficiently produce a proxy for the optimal low-rank approximation.
2.  **Moment Subspace Transformation:** Translates first-order moments between preceding and newly updated subspaces to maintain alignment.
3.  **Low-Rank Steepest Descent:** Employs exact SVD to orthogonalize the moment matrix exactly, ensuring stable update directions.
4.  **Original Space Update:** Incorporates the orthogonal term of the gradient outside the low-rank subspace to maximize information use without extra memory.

---

## 📈 Performance Highlights

### Pre-training LLaMA (C4 Dataset)
SUMO consistently achieves lower validation perplexity with a smaller memory footprint than leading methods.

| Model | GaLore (PPL/Mem) | **SUMO (PPL/Mem)** |
| :--- | :--- | :--- |
| **LLaMA-350M** | 18.95 (1.22G)  | **18.69 (1.16G)**  |
| **LLaMA-1B** | 15.64 (4.38G)  | **14.68 (3.84G)**  |

### Fine-tuning RoBERTa-Base (GLUE Benchmark)
SUMO (rank=8) outperforms GaLore and LoRA across multiple tasks while utilizing significantly less memory.

* **QNLI**: 93.67% 
* **RTE**: 81.37% 
* **MRPC**: 93.7 F1 

---

## 🖋️ Authors
* **Yehonathan Refael** - Tel Aviv University 
* **Guy Smorodinsky** - Ben Gurion University
* **Tom Tirer** - Bar-Ilan University
* **Ofir Lindenbaum** - Bar-Ilan University 
 

---

## 📜 Citing
If you find SUMO helpful in your research, please cite our work:

```bibtex
@article{refael2025sumo,
  title={SUMO: Subspace-Aware Moment-Orthogonalization for Accelerating Memory-Efficient LLM Training},
  author={Refael, Yehonathan and Smorodinsky, Guy and Lindenbaum, Ofir and Tirer, Tom},
  journal={arXiv preprint arXiv:2505.24749},
  year={2025}
}