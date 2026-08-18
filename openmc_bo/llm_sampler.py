"""
llm_sampler.py — LLM-guided Generative Seeding for initial GP training (replaces LHS)
---------------------------------------------------------------------------------------

This is basically a drop-in replacement for the role that lhs_sampler.py currently plays in the pipeline. Instead of using a Latin Hypercube to generate INITIAL_SAMPLES_LHS space-filling points, the LLM is prompted to use its reactor-physics knowledge to directly generate distinct, physically sensible continuous "desirability score" vectors over the n_vars unique symmetric positions. The idea is to let the LLM suggest patterns such as checkerboard alternation, ring-loaded or mid-radius-peaked layouts, diagonal banding, edge-avoiding clusters, etc., rather than having to manually code each of these heuristics.

The LLM does not directly decide which positions are high- or low-enrichment. It only gives each position a score based on how strongly it thinks that position should end up as high-enrichment. The existing RunConfig.decode() / decode_enrichment_vector() pipeline in config.py is left unchanged and handles the actual conversion to the binary enrichment layout. It enforces the required core inventory (n_high_rods) and the hard corner mask using the existing top-d / top-o inventory-matching argmax logic.

This also means that, from the rest of the pipeline's point of view, llm_sampler.py behaves just like lhs_sampler.py. It returns an (n_initial_samples, n_vars) array of continuous values in [cfg.lower, cfg.upper], passes the layouts through the same injected evaluate_fn, and returns the same dictionary structure. So the resulting data can be passed directly into run_bo() in exactly the same way that lhs_data from run_lhs() is used today.

"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import requests
from google import genai
from google.genai import types
from tqdm.notebook import tqdm

from .config import RunConfig
import time

GROK_URL = "https://api.x.ai/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"


@dataclass
class LLMSampleBatch:
    scores: np.ndarray          # (batch_size, n_vars)
    patterns: List[str]
    reasoning: str
    used_fallback: bool
    raw_response: Optional[str] = None


class LLMGridSampler:
    """
    Wraps the LLM call + JSON parsing needed to generate physically-motivated
    initial enrichment-layout score vectors, in batches, for GP seeding.
    """

    def __init__(
        self,
        api_key: str,
        cfg: RunConfig,
        config_path: str = "llmsampler_prompt.json",
        provider: str = "grok",
        timeout_s: float = 60.0,
        raw_log_path: str = "llmsampler_replies_log.txt",
    ):
        if provider not in ("grok", "gemini", "anthropic", "groq", "sarvam"):
            raise ValueError(f"Unknown provider: {provider}")

        self.api_key = api_key
        self.cfg = cfg
        self.provider = provider
        self.timeout_s = timeout_s
        self.raw_log_path = raw_log_path

        with open(config_path, "r") as f:
            self.llm_config = json.load(f)

        self.llm_prompt = self.llm_config["llm_prompt"]
        self.command_template = self.llm_config["command_template"]
        self.defaults = self.llm_config["defaults"]
        self.provider_settings = self.llm_config["provider_settings"][provider]

        if self.provider == "gemini":
            self.gemini_client = genai.Client(api_key=self.api_key)

    
    # Public API
    # ------------------------------------------------------------------ #

    def generate_batch(
        self,
        batch_size: int,
        batch_index: int,
        n_batches: int,
    ) -> LLMSampleBatch:
        """
        Query the LLM for one batch of `batch_size` distinct score vectors.
        Never raises -- on any failure (network, bad JSON, schema violation) it falls
        back to random-uniform scores for the whole batch.
        """
        prompt = self._build_prompt(batch_size, batch_index, n_batches)

        try:
            raw_text = self._query_llm(prompt)
            with open(self.raw_log_path, "a", encoding="utf-8") as f:
                f.write(f"=== BATCH {batch_index + 1}/{n_batches} ===\n")
                f.write(raw_text + "\n\n")
        except Exception as exc:  # network / auth / provider errors
            return self._fallback(batch_size, reason=f"LLM call failed: {exc}")

        try:
            return self._parse_response(raw_text, batch_size)
        except Exception as exc:  # malformed / hallucinated JSON
            return self._fallback(
                batch_size, reason=f"Could not parse LLM response ({exc})",
                raw_response=raw_text,
            )

    
    # Prompt construction
    # ------------------------------------------------------------------ #

    def _build_prompt(self, batch_size: int, batch_index: int, n_batches: int) -> str:
        cfg = self.cfg
        sym_index = cfg.sym_index
        last = cfg.n_rods_side - 1
        corners = {(0, 0), (0, last), (last, 0), (last, last)}

        n_diag = sum(1 for (r, c) in sym_index if r == c)
        n_offdiag = len(sym_index) - n_diag
        corner_indices = [
            i for i, (r, c) in enumerate(sym_index)
            if (r, c) in corners or (c, r) in corners
        ]
        sym_index_map = "  ".join(
            f"{i}:({r},{c}){'D' if r == c else ''}"
            for i, (r, c) in enumerate(sym_index)
        )

        return self.command_template.format(
            n_rods_side=cfg.n_rods_side,
            n_rods_total=cfg.n_rods_total,
            n_vars=cfg.n_vars,
            n_diag=n_diag,
            n_offdiag=n_offdiag,
            corner_indices=corner_indices,
            sym_index_map=sym_index_map,
            n_high_rods=cfg.n_high_rods,
            enr_low=cfg.enr_low,
            enr_high=cfg.enr_high,
            k_target=cfg.k_target,
            ppf_target=cfg.ppf_target,
            batch_index=batch_index + 1,
            n_batches=n_batches,
            batch_size=batch_size,
        )

    
# *************************LLM provider calls**************************************************************************
    

    def _query_llm(self, prompt: str) -> str:
        if self.provider == "anthropic":
            return self._query_anthropic(prompt)
        elif self.provider == "gemini":
            return self._query_gemini(prompt)
        elif self.provider == "grok":
            return self._query_grok(prompt)
        elif self.provider == "groq":
            return self._query_groq(prompt)
        elif self.provider == "sarvam":
            return self._query_sarvam(prompt)

        
    def _query_groq(self, prompt: str) -> str:
        """Queries Groq via its OpenAI-compatible chat-completions endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "model": self.provider_settings["model"],
            "max_tokens": self.provider_settings["max_tokens"],
            "temperature": self.provider_settings["temperature"],
            "messages": [
                {"role": "system", "content": self.llm_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            
            #"response_format": {"type": "json_object"} 
        }
        
        resp = requests.post(
            GROQ_URL,
            headers=headers,
            json=body,
            timeout=self.timeout_s
        )
        
        if not resp.ok:
            raise RuntimeError(
                f"HTTP {resp.status_code} Error from Groq: {resp.text}"
            )
        
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        
        message = choices[0].get("message", {})
        return message.get("content", "") or ""






    def _query_sarvam(self, prompt: str) -> str:
        headers = {
            "api-subscription-key": self.api_key, 
            "Content-Type": "application/json",
        }

        body = {
            "model": self.provider_settings["model"],
            "max_tokens": self.provider_settings["max_tokens"],
            "temperature": self.provider_settings["temperature"],
            "messages": [
                {"role": "system", "content": self.llm_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            
            # Turn off reasoning to prevent the model from returning null.
            "reasoning_effort": None,
            # "response_format": {"type": "json_object"},      –––––––––––––––> for sarvam
        }

        resp = requests.post(
            SARVAM_URL,
            headers=headers,
            json=body,
            timeout=self.timeout_s,
        )

        if not resp.ok:
            raise RuntimeError(
                f"HTTP {resp.status_code} error from Sarvam: {resp.text}"
            )

        data = resp.json()
        choices = data.get("choices", [])
        
        if not choices:
            print(f"\n API Payload: {data}")
            return ""

        message = choices[0].get("message", {})
        
        content = message.get("content")
        
        if content is None:
            print(f"\n[ for debugging - Sarvam gave a null response. message block: {message}")
            return ""
            
        return content
        
        
        
    def _query_anthropic(self, prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.provider_settings["model"],
            "max_tokens": self.provider_settings["max_tokens"],
            "temperature": self.provider_settings["temperature"],
            "system": self.llm_prompt,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post(ANTHROPIC_URL, headers=headers, json=body, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return "".join(
            block.get("text", "") for block in data.get("content", [])
            if block.get("type") == "text"
        )

    def _query_grok(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.provider_settings["model"],
            "max_tokens": self.provider_settings["max_tokens"],
            "temperature": self.provider_settings["temperature"],
            "messages": [
                {"role": "system", "content": self.llm_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        resp = requests.post(GROK_URL, headers=headers, json=body, timeout=self.timeout_s)
        if not resp.ok:
            raise RuntimeError(f"HTTP {resp.status_code} Error from x.ai: {resp.text}")
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""

    def _query_gemini(self, prompt: str) -> str:
        response = self.gemini_client.models.generate_content(
            model=self.provider_settings["model"],
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.llm_prompt,
                temperature=self.provider_settings["temperature"],
                max_output_tokens=self.provider_settings["max_tokens"],
                response_mime_type="application/json",
            )
        )
        if not response.text:
            return ""
        return response.text


# ************************************************************************************************####################
    
    # JSON parsing
    # ------------------------------------------------------------------ #

    def _extract_json_block(self, raw_text: str) -> dict:
        """Cleaned up JSON extractor since Mixtral does not use <think> tags."""
        text = raw_text.strip()

        # Strip standard markdown formatting if present
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
        
        # Grab everything between the outermost curly braces
        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No valid JSON object found in the LLM response.")
            
        json_str = match.group(1)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Extracted string is not valid JSON: {e}")

    
    def _parse_response(self, raw_text: str, batch_size: int) -> LLMSampleBatch:
        payload = self._extract_json_block(raw_text)
        reasoning = str(payload.get("reasoning", "no reasoning given by the llm."))
        samples = payload.get("samples", [])
        if not isinstance(samples, list) or len(samples) == 0:
            raise ValueError("'samples' missing or empty in LLM response")

        n_vars = self.cfg.n_vars
        min_s = self.defaults.get("min_score", 0.0)
        max_s = self.defaults.get("max_score", 1.0)

        rows: List[np.ndarray] = []
        patterns: List[str] = []
        rng = np.random.default_rng()

        for j in range(batch_size):
            if j < len(samples) and isinstance(samples[j], dict):
                raw_scores = samples[j].get("scores", None)
                pattern = str(samples[j].get("pattern", f"llm_sample_{j}"))
            else:
                raw_scores = None
                pattern = f"fallback_random_{j}"

            if raw_scores is None or len(raw_scores) != n_vars:
                # Per-sample fallback -- keeps the batch size guaranteed without
                # discarding otherwise-valid samples elsewhere in the batch.
                row = rng.uniform(min_s, max_s, size=n_vars)
                if raw_scores is not None:
                    pattern = f"fallback_random_{j}(bad_length)"
            else:
                row = np.clip(np.asarray(raw_scores, dtype=float), min_s, max_s)

            rows.append(row)
            patterns.append(pattern)

        return LLMSampleBatch(
            scores=np.vstack(rows), patterns=patterns, reasoning=reasoning,
            used_fallback=False, raw_response=raw_text,
        )


   
    
    def _fallback(
        self, batch_size: int, reason: str, raw_response: Optional[str] = None,
    ) -> LLMSampleBatch:
        min_s = self.defaults.get("min_score", 0.0)
        max_s = self.defaults.get("max_score", 1.0)
        rng = np.random.default_rng()
        rows = rng.uniform(min_s, max_s, size=(batch_size, self.cfg.n_vars))
        return LLMSampleBatch(
            scores=rows,
            patterns=[f"fallback_random_{j}" for j in range(batch_size)],
            reasoning=f"[fallback] {reason}",
            used_fallback=True,
            raw_response=raw_response,
        )



# Sample generation (just like in lhs_sampler.py)
# ------------------------------------------------------------------------ #

def generate_llm_samples(
    cfg: RunConfig,
    api_key: str,
    config_path: str = "llmsampler_prompt.json",
    provider: str = "grok",
    batch_size: Optional[int] = None,
    timeout_s: float = 60.0,
    raw_log_path: str = "llmsampler_replies_log.txt",
    verbose: bool = True,
) -> np.ndarray:
    """
    Generate cfg.n_initial_samples distinct, LLM-proposed continuous score vectors, batched to keep each LLM call's output small and reliable. Returns an array of
    shape (n_initial_samples, n_vars) scaled into [cfg.lower, cfg.upper] -- identical shape/scale contract to lhs_sampler.generate_lhs_samples.
    """
    sampler = LLMGridSampler(
        api_key=api_key, cfg=cfg, config_path=config_path, provider=provider,
        timeout_s=timeout_s, raw_log_path=raw_log_path,
    )

    n_total = cfg.n_initial_samples
    bs = batch_size or sampler.defaults.get("default_batch_size", 6)
    n_batches = int(np.ceil(n_total / bs))

    rows: List[np.ndarray] = []
    seen_grids = set()
    rng = np.random.default_rng(cfg.random_seed)
    n_fallback_batches = 0

    for b in tqdm(range(n_batches), desc="LLM sample batches", disable=not verbose):
        this_bs = min(bs, n_total - len(rows))
        batch = sampler.generate_batch(this_bs, b, n_batches)
        if batch.used_fallback:
            n_fallback_batches += 1

        for row, pattern in zip(batch.scores, batch.patterns):
            grid = cfg.decode(row)
            grid_key = tuple(np.round(grid, 3).ravel().tolist())
            tries = 0
            while grid_key in seen_grids and tries < 3:
                row = np.clip(row + rng.normal(0, 0.05, size=row.shape), 0.0, 1.0)
                grid = cfg.decode(row)
                grid_key = tuple(np.round(grid, 3).ravel().tolist())
                tries += 1
            seen_grids.add(grid_key)
            rows.append(row)

        if verbose:
            print(f"  [batch {b + 1}/{n_batches}] {len(batch.scores)} samples  "
                  f"({'FALLBACK' if batch.used_fallback else 'LLM'})  "
                  f"reasoning: {batch.reasoning}")

            
    # ********* DELAY BLOCK ***************
        # Pause for 12 seconds between batches to align with Groq rate limits, but don't sleep after the very last batch.
        if b < n_batches - 1:
            time.sleep(20.0)   # increased from 12 ––> 20 for sarvam. 

    X_unit = np.vstack(rows)[:n_total]
    X = cfg.lower + X_unit * (cfg.upper - cfg.lower)

    if verbose and n_fallback_batches:
        print(f"  NOTE: {n_fallback_batches}/{n_batches} batches used the random-uniform "
              f"fallback (LLM call failed or returned corrupted JSON).")

    return X


def run_llmsampler(
    cfg: RunConfig,
    evaluate_fn: Callable[[np.ndarray], dict],
    api_key: str,
    config_path: str = "llmsampler_prompt.json",
    provider: str = "grok",
    batch_size: Optional[int] = None,
    timeout_s: float = 60.0,
    raw_log_path: str = "llmsampler_replies_log.txt",
    verbose: bool = True,
) -> dict:
    """
    LLM-generative-seeding drop-in replacement for lhs_sampler.run_lhs().

    """
    X = generate_llm_samples(
        cfg, api_key=api_key, config_path=config_path, provider=provider,
        batch_size=batch_size, timeout_s=timeout_s, raw_log_path=raw_log_path,
        verbose=verbose,
    )

    keff_list, keff_std_list, ppf_list = [], [], []
    score_list, base_score_list, adj_list = [], [], []

    if verbose:
        avg_enr = (
            cfg.n_high_rods * cfg.enr_high
            + (cfg.n_rods_total - cfg.n_high_rods) * cfg.enr_low
        ) / cfg.n_rods_total
        print(f"\nEvaluating {cfg.n_initial_samples} LLM-generated samples  "
              f"[{cfg.n_rods_side}×{cfg.n_rods_side}, "
              f"{cfg.n_vars}-D continuous relaxation]  (provider={provider})")
        print(f"  ENR inventory : {cfg.n_high_rods} × {cfg.enr_high}%  +  "
              f"{cfg.n_rods_total - cfg.n_high_rods} × {cfg.enr_low}%  "
              f"=> avg {avg_enr:.4f} wt%")
        print(f"  k∞ target     : {cfg.k_target}   "
              f"PPF limit ≤ {cfg.ppf_target}")

    for i, x in enumerate(tqdm(X, desc="LLM samples", disable=not verbose)):
        res = evaluate_fn(x)
        k = float(res["keff"])
        kstd = float(res.get("keff_std", 0.0))
        ppf = float(res.get("ppf", 0.0))
        adj = float(res.get("adj_high_rods", 0))
        base_s = cfg.composite_score(k, ppf)

        #*************** Applies adjacency penalty using the count injected by the Environment************#########
        s = base_s
        if getattr(cfg, "use_adjacency_penalty", False):
            s = base_s - cfg.adj_penalty_weight * adj
        #********************************************##############

        score_list.append(s)
        base_score_list.append(base_s)
        adj_list.append(adj)
        keff_list.append(k)
        keff_std_list.append(kstd)
        ppf_list.append(ppf)

        if verbose:
            _print_sample(cfg, i, k, kstd, ppf, s)

    keff = np.array(keff_list)
    kstd = np.array(keff_std_list)
    ppf = np.array(ppf_list)
    score = np.array(score_list)
    base_score = np.array(base_score_list)
    adj_high_rods = np.array(adj_list)

    if verbose:
        _print_llm_summary(cfg, keff, ppf, score)

    return dict(
        X=X, keff=keff, keff_std=kstd, ppf=ppf, score=score,
        base_score=base_score, adj_high_rods=adj_high_rods,
    )


#************* Helpers *****************############

def _print_sample(cfg, i, keff, keff_std, ppf, score):
    idx_str = f"[{i+1:2d}/{cfg.n_initial_samples}]"
    ppf_str = f"  PPF={ppf:.3f}" if cfg.ppf_target is not None else ""
    std_str = f" ±{keff_std:.5f}" if keff_std > 0 else ""
    print(f"  {idx_str}  k∞={keff:.5f}{std_str}{ppf_str}  score={score:.5f}")


def _print_llm_summary(cfg, keff, ppf, score):
    n = len(keff)
    print(f"\n── LLM sampler summary {'─'*38}")
    print(f"Samples          : {n}")
    print(f"k∞ range         : [{keff.min():.5f}, {keff.max():.5f}]")
    print(f"k∞ target        : {cfg.k_target}")
    if cfg.ppf_target is not None:
        feasible = (ppf <= cfg.ppf_target).sum()
        print(f"PPF range        : [{ppf.min():.3f}, {ppf.max():.3f}]")
        print(f"Feasible         : {feasible} / {n}")
    best_i = int(np.argmax(score))
    print(f"Best score       : {score.max():.5f}  "
          f"(k∞={keff[best_i]:.5f}, "
          f"|Δk|={abs(cfg.k_target - keff[best_i]):.5f})")