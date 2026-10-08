# One-Week PINN Study Plan and Holiday Review Roadmap
## Goal
The goal for **Monday, 5 October through Sunday, 11 October 2026** is not to master every derivation in all eight papers. With 1–2 focused hours per day, the realistic target is to **understand and summarize all eight papers, study three or four central papers more deeply, and finish one trustworthy baseline PINN notebook**. This matches the user's current stage: new to PINNs but already able to implement an ODE residual, initial condition, automatic differentiation, and a combined loss.[^1]

By Sunday evening, the required outputs are:

- A working logistic-equation PINN notebook with analytical and `solve_ivp` comparisons.
- Eight short paper records, one per paper.
- Three or four deeper paper summaries.
- A one-page concept map of PINNs.
- A title, research question, and detailed outline for a short holiday review on PINNs in plasma and fusion.

The original PINN framework treats neural networks as differentiable approximations constrained by governing equations and covers both forward solution and inverse/discovery problems. That distinction should organize both the week's learning and the eventual review.[^2][^3]
## Weekly timetable
| Day | Time | Main task | Concrete actions | Deliverable |
|---|---:|---|---|---|
| **Mon 5 Oct** | 75–90 min | Establish the PINN foundation | Re-run the logistic ODE notebook; label every object as network, data point, collocation point, residual, initial-condition loss, or data loss; verify the numerical and analytical solutions use the same equation and initial value. | Clean notebook plus a half-page explanation of the training loop. |
| **Tue 6 Oct** | 90 min | Test the baseline rather than merely running it | Train with 10, 50, 100, and 200 collocation points; record PDE, boundary, and data losses separately; evaluate residual on a denser unseen grid; save one comparison plot. | Small experiment table and one final plot. |
| **Wed 7 Oct** | 90–120 min | Read the easiest two papers | Use the three-pass method below; focus on problem, inputs/outputs, governing physics, loss, and evidence; do not chase every equation. | Two one-page paper cards. |
| **Thu 8 Oct** | 90–120 min | Read two medium papers | Identify whether each work is a true PINN, a physics-constrained surrogate, or an ordinary surrogate; map each loss term to a physical or data constraint. | Two paper cards plus a method-comparison row for each. |
| **Fri 9 Oct** | 90–120 min | Deep-read one central plasma/fusion PINN paper | Reconstruct its pipeline: physical problem → equations → network → training points/data → loss → validation → limitations. Inspect one central figure and explain what it actually proves. | One two-page deep summary and one reproduced schematic drawn by hand or digitally. |
| **Sat 10 Oct** | 2–3 h in two blocks | Cover the remaining papers and synthesize | Block 1: rapid-read the remaining papers. Block 2: fill the cross-paper matrix and choose the three or four papers that deserve deep study during the holidays. | All eight paper cards completed; full comparison matrix. |
| **Sun 11 Oct** | 2–3 h in two blocks | Consolidate and design the review | Block 1: explain PINNs aloud without notes for five minutes, then fix gaps. Block 2: define review scope, draft title, research question, section outline, and assign papers to sections. | One-page PINN map and 1–2 page review outline. |

If only one hour is available on a weekday, stop after the listed deliverable rather than extending into optional reading. Saturday and Sunday should absorb unfinished paper cards; the logistic notebook and review outline are the two non-negotiable outputs.
## Monday coding checklist
The notebook should solve the normalized logistic problem

\[
\frac{df}{dt}=R f(1-f), \qquad f(0)=f_0, \qquad 0<f_0<1.
\]

Use the same values of `R`, `f0`, and `domain` everywhere: analytical solution, `solve_ivp`, training data, boundary loss, and PDE residual. The final residual must be formed from the network value, not from the model object:

```python
f_t = model(t)
df_dt = df(model, t)
residual = df_dt - R * f_t * (1.0 - f_t)
```

Finish Monday only when these checks pass:

- `x_train`, `y_train`, `t`, and `model(...)` use shape `(N, 1)`.
- `t.requires_grad` is `True`.
- The initial condition uses `f0`, not a variable ambiguously named `t0`.
- The analytical solution, numerical solution, and PINN solve exactly the same initial-value problem.
- Evaluation uses a dense grid that is separate from the collocation grid.
## Tuesday experiment
Create a compact table with these columns:

| Run | Collocation points | Epochs | Learning rate | PDE loss | IC loss | Data loss | Dense-grid relative error |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 10 | fixed | fixed | record | record | record | record |
| B | 50 | fixed | fixed | record | record | record | record |
| C | 100 | fixed | fixed | record | record | record | record |
| D | 200 | fixed | fixed | record | record | record | record |

Keep all settings except the number of collocation points unchanged. Evaluate the residual on at least 500 new points rather than only at the training collocation points; a small residual only on sampled points does not establish that the differential equation is satisfied between them. The distinction between training residual and unseen-grid residual is especially important because PINNs can fit collocation points without learning a uniformly accurate solution.[^4]
## Paper-reading method
Use three passes for every paper. The aim is to extract its scientific argument, not to understand every line on first contact.
### Pass 1: Ten minutes
Read only the title, abstract, introduction's final paragraph, figures, captions, conclusion, and section headings. Write one sentence for each:

- What problem is solved?
- Why is a PINN or neural surrogate used?
- What is the main claimed result?
- Is the work directly relevant to plasma or fusion?
### Pass 2: Twenty minutes
Locate and record:

- Governing ODE/PDE and dependent variables.
- Forward, inverse, reconstruction, parameter-identification, or surrogate task.
- Network inputs and outputs.
- Data, initial/boundary conditions, and collocation points.
- Every term in the total loss.
- Baseline or reference solver.
- Primary error metric.
### Pass 3: Twenty to thirty minutes
Use this only for the selected three or four deep papers. Trace one complete path from an input point through the network, automatic differentiation, residual calculation, optimization, and validation. Then inspect whether the experiments support the stated claim and write at least one limitation.
## Paper card template
Use exactly one page per paper:

```markdown
# Paper title

- Type: PINN / physics-constrained surrogate / ordinary surrogate
- Plasma problem:
- Research question:
- Governing equation(s):
- Forward or inverse problem:
- Network input(s):
- Network output(s):
- Physics information:
- Data used:
- Total loss terms:
- Reference method:
- Main result:
- Strongest figure/table:
- Main limitation:
- Relevance to the review: high / medium / low
- One thing still unclear:
- Three-sentence summary:
```

A paper counts as “understood” when this card can be completed without copying the abstract. Unknown items should be marked clearly rather than guessed.
## Reading priority
Retain the earlier easiest-to-hardest ranking, but classify the papers into three working groups:

| Group | This week's depth | Purpose |
|---|---|---|
| Introductory or clear application papers | Deep-read 1–2 | Learn the standard PINN pipeline and vocabulary. |
| Plasma/fusion PINN papers | Deep-read 2 | Supply the core evidence for the holiday review. |
| Advanced methods, singular perturbations, or difficult mathematical variants | Structural read only | Identify motivation, modification to vanilla PINNs, results, and limitations; postpone full derivations. |
| ICRF or other non-PINN surrogate papers | Comparative read | Clarify the difference between data-driven surrogate modeling and enforcing equations in the loss. |

The singular-perturbation boundary-layer paper should remain late in the sequence because it adds asymptotic structure and coupled approximations to the standard PINN machinery. The PINNeik paper is useful for asking when a PINN should be compared with a classical solver, while the ICRF surrogate papers are useful contrasts rather than foundational PINN texts.[^1][^5]
## Concept map
The Sunday concept map should contain these connected boxes:

1. **Problem definition:** domain, variables, parameters, governing equations.
2. **Approximation:** neural network \(u_\theta(x,t,\mu)\).
3. **Automatic differentiation:** spatial and temporal derivatives.
4. **Residual:** governing equation evaluated with the network.
5. **Constraints:** initial conditions, boundary conditions, observations, and integral/conservation constraints.
6. **Loss:** weighted sum of residual and constraint errors.
7. **Optimization:** Adam and, where used, L-BFGS.
8. **Validation:** reference solver, unseen points, residual field, conservation, and parameter error.
9. **Failure modes:** loss imbalance, insufficient sampling, poor scaling, stiffness, spectral bias, and convergence to misleading low-loss solutions.

The classic formulation covers both data-driven solution and data-driven discovery, while later work extends PINNs through improved architectures, sampling, optimization, and physics integration.[^2][^6]
## Review-paper scope
A manageable holiday review topic is:

**“Physics-Informed Neural Networks in Magnetic-Confinement Fusion: Applications, Advantages, and Training Challenges.”**

A focused research question is:

> How are PINNs currently used in magnetic-confinement fusion, what advantages do they offer for inverse and data-sparse problems, and what prevents routine deployment?

This scope is better than a general review of all PINNs because it matches the user's laboratory direction and keeps the literature manageable. Current fusion applications include tokamak equilibrium reconstruction from multiple diagnostics, transport modeling, turbulence-field inference, MHD evolution, and kinetic or adjoint problems.[^7][^8][^9][^10][^11]
## Review outline
### Introduction
- Why fusion modeling combines expensive simulation, sparse diagnostics, inverse problems, and physical constraints.
- Definition of PINNs and difference from ordinary neural surrogates.
- Scope: magnetic-confinement fusion rather than every plasma application.
### PINN methodology
- Network approximation and automatic differentiation.
- PDE residual, boundary/initial conditions, and observation loss.
- Forward versus inverse problems.
- Validation against analytical, numerical, synthetic, or experimental references.
### Fusion applications
Organize by scientific task rather than paper-by-paper chronology:

- **Equilibrium reconstruction:** infer internal plasma fields from magnetic and other diagnostic measurements; recent work incorporates Thomson scattering and interferometer-polarimetry.[^7][^12]
- **Transport and turbulence:** solve transport systems or infer missing fields from partial and noisy measurements.[^9][^11]
- **Disruptions and MHD:** learn time-dependent quasi-static MHD behavior in axisymmetric tokamak geometry.[^13][^8]
- **Kinetic and energetic-particle problems:** learn adjoint or kinetic quantities such as runaway-electron probability or energetic-particle escape.[^14][^10]
### Advantages
Discuss advantages only where supported by a specific task:

- Combining sparse/noisy diagnostics with governing equations.
- Solving inverse problems and estimating unobserved fields or parameters.
- Producing differentiable continuous approximations.
- Repeated evaluation after training and potential use as a surrogate.
- Flexible constraints and mesh-free sampling.

Avoid claiming that PINNs are universally faster or more accurate than classical solvers. Some fusion work reports promising reconstruction or surrogate behavior, but training can remain scenario-specific and take hours, limiting immediate real-time use.[^15]
### Limitations
- Difficult optimization and competing loss terms.
- Sensitivity to scaling, sampling, architecture, and random initialization.
- Computational cost of repeated automatic differentiation.
- Weak guarantees outside sampled points or training regimes.
- Need for comparisons with established solvers and experimental validation.

A useful critical contrast is that longer training can improve quantitative agreement in tokamak MHD examples, while equilibrium-reconstruction studies explore pretraining to reduce convergence time.[^8][^16]
### Research gaps
- Standardized benchmarks against established plasma codes.
- Reliable uncertainty quantification.
- Transfer across discharges, devices, and parameter regimes.
- Robust training for multiscale, stiff, turbulent, and kinetic systems.
- Integration with experimental diagnostics and real-time workflows.
### Conclusion
State where PINNs appear most credible: physics-constrained inverse problems, sparse-data reconstruction, and reusable parametric representations. State where evidence is still insufficient: replacement of mature high-fidelity solvers for routine forward simulation.
## Holiday writing schedule
| Session | Duration | Task | Output |
|---|---:|---|---|
| 1 | 2 h | Finalize scope and inclusion criteria; sort the eight papers by application. | Locked title, question, and paper matrix. |
| 2 | 2 h | Search for foundational and recent fusion PINN papers; use citation chaining. | Curated source list. |
| 3 | 2 h | Write methodology section from the logistic notebook and foundational paper. | 600–800 words. |
| 4 | 2 h | Write equilibrium and reconstruction applications. | 600–800 words. |
| 5 | 2 h | Write transport, turbulence, MHD, and kinetic applications. | 800–1,000 words. |
| 6 | 2 h | Write advantages, limitations, and comparison with classical solvers. | 700–900 words. |
| 7 | 2 h | Write introduction, abstract, and conclusion last. | Complete first draft. |
| 8 | 2 h | Verify every technical claim, equation, caption, and citation; remove repetition. | Submission-ready short review. |

A realistic final length is approximately 3,000–4,000 words plus one taxonomy table and one conceptual figure. The paper should synthesize applications and limitations rather than provide eight isolated summaries.
## Daily completion rule
End each study session with three lines:

1. **Today:** one concept that became clear.
2. **Evidence:** one equation, figure, experiment, or code result that supports it.
3. **Next:** one unresolved question to begin with tomorrow.

Use a 50-minute focus block, a 10-minute break, and a final 20–30-minute synthesis block. Reading counts only when it produces a paper card, comparison-matrix entry, code test, or review paragraph.

---

## References

1. [Hi... Now I have a new lab, and there people work with PINN. And i have never worked with this, whatever it is. But now I have to figure PINN out... 
I will send you like a few article from my supervisor. I want you to arrange them from hardest to eseast one. If i have mistake in my typing, please correct me and explain mistakes.](https://www.perplexity.ai/search/e0a94744-b66d-4b38-bfb5-020d131c6a18) - I reviewed all eight papers. For someone with your plasma-physics and scientific-computing backgroun...

2. [Authors | Physics Informed Deep Learning - Maziar Raissi](https://maziarraissi.github.io/PINNs/) - Data-driven solutions and discovery of Nonlinear Partial Differential Equations

3. [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://www.aer.com/siteassets/21data/raissietal2019.pdf)

4. [[Literature Review] PINNs Failure Modes are Overfitting](https://www.themoonlight.io/en/review/pinns-failure-modes-are-overfitting) - This paper argues that these failure modes are fundamentally caused by overfitting to the collocatio...

5. [Can you show results, that explains advanteges of PINN compare to classic PDE solvers from papaers, that I have sent you before?](https://www.perplexity.ai/search/b63d412e-bd26-447c-93e9-cd7eb36233a9) - Yes—but the papers do not demonstrate that PINNs are universally better than classical PDE solvers. ...

6. [Physics-Informed Neural Networks and Extensions - arXiv.org](https://arxiv.org/html/2408.16806)

7. [Physics-informed neural networks for the modelling of ...](https://art.torvergata.it/retrieve/e5e94c28-eb4d-4108-84ea-825ee52787ea/Rutigliano+et+al_2025_Plasma_Phys._Control._Fusion_10.1088_1361-6587_addde6.pdf)

8. [A Physics-Informed Neural Network for Solving the Quasi-static ...](https://arxiv.org/html/2604.20085v1)

9. [Leveraging Physics-Informed Neural Computing for Transport Simulations of Nuclear Fusion Plasmas](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4554149) - For decades, plasma transport simulations in tokamaks have used the finite difference method (FDM) t...

10. [A physics-constrained deep learning surrogate model of the ... - OSTI](https://www.osti.gov/servlets/purl/2568830)

11. [Leveraging turbulence data with physics informed neural networks](https://arxiv.org/html/2412.20130v1)

12. [[PDF] Physics-informed neural networks for the modelling of interferometer ...](https://art.torvergata.it/bitstream/2108/425168/1/Rutigliano+et+al_2025_Plasma_Phys._Control._Fusion_10.1088_1361-6587_addde6.pdf)

13. [A Physics-Informed Neural Network for Solving the Quasi-static ...](https://papers.cool/arxiv/2604.20085) - A physics-informed neural network (PINN) is developed, for the first time, to learn the time-depende...

14. [An Adjoint Formulation of Energetic Particle Confinement](https://arxiv.org/pdf/2511.11968v2.pdf)

15. [Physics-Informed Neural Networks for Multi-Diagnostic ...](https://conferences.iaea.org/event/393/contributions/36675/attachments/21793/37466/Rutigliano_IAEA.pdf)

16. [Optimisation of physics-informed neural network architecture ...](https://art.torvergata.it/retrieve/97c859c8-19aa-41e0-9e19-c2e523f16dfa/Rutigliano_2026_Plasma_Phys._Control._Fusion_68_045002.pdf)

