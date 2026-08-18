# OpenMC-LOBOS

**L**LM-**O**rchestrated **B**ayesian **O**ptimization with **S**eeding for BWR fuel-assembly enrichment layouts.


OpenMC-LOBOS is a sibling project to [OpenMC-PRISM](https://github.com/light0vij/OpenMC-PRISM). It keeps the same physics model, symmetry decoding, GP-surrogate/Expected-Improvement Bayesian Optimisation (BO) loop, and heuristic constraints used in OpenMC-PRISM.

The main change is in **how the BO process is initialised and guided**. Instead of relying only on Latin Hypercube Sampling (LHS) to generate space-filling points without any physics-based preference, OpenMC-LOBOS introduces an LLM at two stages of the optimisation. First, it generates physically motivated initial layouts, and second, it adjusts the search bounds during the BO iterations.

The underlying optimisation and physics workflow remains unchanged. The LLM is not replacing OpenMC, the GP surrogate, or Expected Improvement. It is only used to influence where the optimisation starts and how the search region is adjusted as the run progresses.


---

## Repository layout

```
openmc-lobos/
│
├── openmc_bo/                  ← Optimiser package — model-agnostic, never edit for new models
│   ├── __init__.py
│   ├── config.py                ← RunConfig dataclass: all tunable BO parameters & penalty weights
│   ├── llm_sampler.py           ← LLM Generative Seeding — drop-in replacement for LHS initial design
│   ├── llm_orchestrator.py      ← LLM Bounds Orchestration — reshapes the EI search box each iteration
│   ├── gp_surrogate.py          ← Matérn-5/2 GP + Expected Improvement acquisition
│   ├── llm_bo_loop.py           ← Bayesian Optimisation loop (pure algorithm, zero environment imports)
│   └── results.py               ← CSV export, convergence plots, design report
├── environments/                ← Physics layer — the OpenMC model lives here
│   ├── __init__.py
│   ├── env.py                   ← OpenMC BWR NxN model; exposes evaluate_for_bo
│   ├── heuristics.py            ← Physics constraint counters (corner violations, adjacency pairs)
│   ├── mapping.py                ← Symmetry mapping and enrichment grid utilities
│   ├── bwr_vis.py                ← Assembly visualisation helpers
│   └── bwr_report.py             ← Per-run design report generation
├── Notebooks/
│   └── LLM_sampled_BO_BWR_NxN.ipynb   ← Main entry point — LLM seeding + LLM bounds orchestration + BO
├── sampler_prompts/              ← LLM Generative Seeding prompt configs (the "how should the LLM seed" knob)
│   ├── disperse_llmsampler.json
│   ├── E_repulsion_llmsampler.json
│   └── hybrid_E_disperse_llmsampler.json
├── prompt2.json                  ← LLM Bounds Orchestration prompt config
├── Results/                      ← Pre-existing sample runs, shipped in the image; new runs land here too
├── Dockerfile                    ← Builds the ghcr.io/light0vij/openmc-lobos image
├── docker-compose.yml            ← Runs JupyterLab locally on port 8888
├── executedownload.sh            ← Downloads ENDF/B-VIII.0 nuclear data on first container start
├── requirements.txt              ← Python dependencies baked into the image
├── .dockerignore
├── .gitignore
└── README.md
```


---

## Overview: The LLM in the loop

The core GP/EI/OpenMC optimisation process remains the same as the one used in OpenMC-PRISM's `BO_Optimisation_BWR_NxN.ipynb`. This project does not change or reimplement the Bayesian Optimisation loop itself. Instead, the LLM is introduced at two specific points in the existing workflow.

### 1. LLM Generative Seeding (`openmc_bo/llm_sampler.py`)

The first use of the LLM is to generate the initial samples used to train the GP surrogate. This replaces the normal LHS-based initial population.

Instead of generating `n_initial_samples` using blind space-filling samples, the LLM is prompted to act as a BWR core-physics engineer and generate batches of continuous desirability scores. Each score corresponds to one unique symmetric rod position in the assembly.

The scores are intended to reflect different loading strategies, such as ring loading, dispersed high-enrichment islands, checkerboard-like arrangements, or edge-focused and edge-avoiding distributions.

The LLM does not directly decide which rods are assigned high or low enrichment. That part of the process is still handled by the existing `RunConfig.decode()` function. The generated scores are ranked, and the decoder assigns exactly `n_high_rods` positions to high enrichment while respecting the corner mask. This is the same decoding process used in the original LHS-based workflow.

From the rest of the BO pipeline's perspective, `run_llmsampler()` behaves the same way as `run_lhs()` and returns the same dictionary structure. This makes the LLM sampler a direct replacement for the original LHS initialisation step without requiring changes to the GP, EI, decoding, or OpenMC evaluation code.

### 2. LLM Bounds Orchestration (`openmc_bo/llm_orchestrator.py`)

The second use of the LLM happens during the BO iterations.

After each OpenMC evaluation, the LLM is given information about the latest design, including k∞, PPF, the number of adjacent high-enrichment rods, and the current predictive uncertainty from the GP surrogate. Based on this information, it recommends how the lower and upper bounds of the search space should be adjusted for the next iteration.

The LLM does not select the next design point itself. Expected Improvement still performs that task. The LLM only recommends the region of the design space that EI should search within by narrowing, widening, or re-centring the active bounds.

In other words, the optimisation logic remains with the existing BO framework. The LLM influences the search region, but EI is still responsible for selecting the actual next candidate.

### Error handling and validation

Both LLM components use a strict JSON response format. If a response is malformed, incomplete, or missing required fields, the optimisation does not fail.

For the sampler, the code falls back to generating a random uniform sample. For the bounds orchestrator, it falls back to the previous bounds or the default configuration. Any fallback is recorded in the logs.

The original LLM responses are also saved to disk for later inspection and debugging:

* `*_sampler_replies.txt`
* `*_rawOrchestrator_replies.txt`

This also makes it possible to inspect what the model actually returned, rather than relying only on the parsed values used by the optimisation.

At no point does the LLM directly modify OpenMC, the GP surrogate, or the Expected Improvement implementation.

### LLM for seeding instead of LHS

The main reason is that LHS treats the initial samples as a purely mathematical space-filling problem. The LLM sampler allows the initial population to be biased towards regions that are already physically reasonable.

For example, the prompt can encourage strategies based on spatial dispersion, avoiding excessive clustering, reducing power peaking, or accounting for ideas such as self-shielding. This means the GP is trained initially on designs that are more likely to be useful, rather than starting entirely from uniformly distributed samples across the available continuous space.

Another advantage is that the sampling behaviour can be changed without modifying the optimiser itself. The different prompt files in `sampler_prompts/*.json` describe different placement strategies using natural language. Changing the prompt can therefore change where the initial GP training data is concentrated without changing the BO code, decoder, or OpenMC model.

The implementation also supports multiple LLM providers, including `grok`, `gemini`, `groq`, `anthropic`, and `sarvam`, using the same JSON response contract. Switching providers is therefore mainly a configuration change rather than requiring separate implementations of the sampler or orchestrator.

There is also an obvious trade-off. An LLM-based sampler is slower and less reproducible than using `scipy.stats.qmc.LatinHypercube`. Each batch requires an API request and response, and the implementation includes `time.sleep()` between batches to handle rate limits. The generated samples can also vary between runs depending on the provider and model settings.

For the 6×6 assembly tested in this project, the BO iterations themselves were strong enough to reduce most of the differences between the different seeding strategies by the end of the optimisation. The final solutions obtained using LLM-generated seeds and the original LHS baseline were therefore very similar. The main difference was more visible at the start of the optimisation, where the LLM-generated samples were already concentrated in narrower, physically motivated regions of the design space.

This approach is likely to be more useful when the search space becomes larger or when the number of available OpenMC evaluations is limited. In those cases, starting BO with a more informed initial population may matter more because the optimiser has fewer iterations available to recover from a poor set of initial samples.

---


## Results summary

Two LLM-based sampling strategies were tested on the same 6×6 BWR fuel assembly containing 23 high-enrichment rods out of 36. `Sampler_2` uses an E-repulsion strategy, while `Sampler_3` combines E-repulsion with explicit centre exclusion. Both were compared against the original plain LHS-seeded BO baseline using identical optimisation settings: 36 initial samples followed by 30 BO iterations.

The final designs were re-evaluated using a 50,000-particle high-fidelity OpenMC simulation.

| Optimisation method                               |      k∞ | σ(k∞) [pcm] |    PPF |
| ------------------------------------------------- | ------: | ----------: | -----: |
| Plain LHS + BO *(baseline)*                       | 1.27174 |       29.79 | 1.1049 |
| `Sampler_2` — E-repulsion seeding                 | 1.27212 |       29.62 | 1.0994 |
| `Sampler_3` — Hybrid repulsion + centre exclusion | 1.27111 |       33.19 | 1.0986 |

*k∞ target = 1.25, PPF limit ≤ 1.30. All values shown above are from the 50,000-particle HiFi re-evaluation.*

For this 6×6 assembly and optimisation budget, all three approaches converged to very similar solutions. Both LLM-seeded runs achieved slightly lower PPF values than the plain LHS baseline, while `Sampler_2` produced the k∞ value closest to the target.

The more interesting result is what happens before BO starts. The initial LLM-generated batches were already concentrated in a narrow region with near-target k∞ values and low PPF values. This happened because the sampling prompts encouraged physically reasonable spatial distributions of the high-enrichment rods rather than generating a completely uninformed initial population.

Changing only the wording of the sampling prompt — for example, using pure electrostatic-style repulsion versus repulsion combined with strict centre exclusion — changed the placement logic and reasoning produced by the LLM without changing the underlying OpenMC, decoding, or BO pipeline.

For this relatively small problem, BO was able to reduce most of the differences between the initial sampling strategies by the end of the run. However, the results show that the sampling prompt can act as a natural-language way of controlling where the initial population, and therefore the initial GP training data, is concentrated. This could become more useful for larger assemblies or optimisation problems with tighter evaluation budgets, where BO has less opportunity to recover from a poor initial population.

Full artefacts for both runs are available in `Results/`.


---



## Installation

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running.


```bash
docker pull ghcr.io/light0vij/openmc-lobos:latest
```

```bash
docker run -p 8888:8888 -v /YOUR_FOLDER_PATH:/nuclear_data ghcr.io/light0vij/openmc-lobos:latest
```

or, to also persist/edit notebooks on the host:

```bash
docker run -p 8888:8888 \
  -v /YOUR_FOLDER_PATH:/nuclear_data \
  -v /YOUR_NOTEBOOKS_PATH:/workspace/Notebooks \
  ghcr.io/light0vij/openmc-lobos:latest
```

Or with `docker compose` (mounts `Notebooks/`, `environments/`, `openmc_bo/`, `sampler_prompts/`, `prompt2.json`,
`Results/`, and a named `nuclear_data` volume — see `docker-compose.yml`):

```bash
docker compose up
```

Then open **http://localhost:8888** — no token or password required.

### LLM API key

The notebook reads its key from the `LLM_API_KEY` environment variable — it is **not** hardcoded in the notebook.
Set it before starting the container:

```bash
docker run -p 8888:8888 \
  -v /YOUR_FOLDER_PATH:/nuclear_data \
  -e LLM_API_KEY=your_key_here \
  ghcr.io/light0vij/openmc-lobos:latest
```

or with compose, either `LLM_API_KEY=your_key_here docker compose up`, or a local `.env` file (already
git-ignored) containing `LLM_API_KEY=your_key_here`.


### Nuclear data

Nuclear data is **not** baked into the image. The ENDF/B-VIII.0 cross-section library (~2 GB), depletion chain, and
Windowed Multipole library are downloaded at container startup by `executedownload.sh` into the folder mounted at
`/nuclear_data`. On first run, `executedownload.sh` checks for `/nuclear_data/cross_sections.xml`; if it's missing,
it downloads and unpacks everything there. On every subsequent run against the same mounted folder, the script
detects the data is already present and skips straight to starting JupyterLab — so startup is near-instant after
the first time.

### Why the image builds fast

OpenMC is installed via [shimwell's pre-built wheels](https://shimwell.github.io/wheels) rather than compiled from
source, which saves roughly 30 minutes of build time per image build compared to a source build.

---

## Notes on assembly size

`RunConfig(n_rods_side=N, n_high_rods=...)` generalises to any N×N assembly under 1/2-diagonal symmetry — the
results above are for the 6×6 case only. The notebook scales `INITIAL_SAMPLES_LHS` (LLM seed count) and
`BO_ITERATIONS` with `(BWR_N / 6) ** 2` so larger assemblies get proportionally more seed samples and BO iterations.

## References

Romano, P. K., Horelik, N. E., Herman, B. R., Nelson, A. G., Forget, B., & Smith, K. (2015). OpenMC: A State-of-the-Art Monte Carlo Code for Research and Development. *Annals of Nuclear Energy*, 82, 90–97. [https://doi.org/10.1016/j.anucene.2014.07.048](https://doi.org/10.1016/j.anucene.2014.07.048)


