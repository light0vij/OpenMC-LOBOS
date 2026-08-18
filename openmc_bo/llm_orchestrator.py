"""
llm_orchestrator.py 
-------------------------



    - Give the LLM the latest OpenMC results (k_inf, PPF, enr_grid) along with the current GP uncertainty.
    - Ask it to recommend the lower and upper bounds for the next BO search.
    - The LLM does not choose the next design itself. The actual candidate is still selected by expected_improvement() in gp_surrogate.py.
    - Keep the LLM response small and validate it before using it.
    - If the response is malformed or contains hallucinated values, do not let it crash the BO loop.
    - In that case, fall back to the previous bounds or the config defaults and continue the optimisation.
    
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests
from google import genai
from google.genai import types

from .config import RunConfig

#from huggingface_hub import InferenceClient #––––––––––––––-> for hugging face/atomicGPT


GROK_URL = "https://api.x.ai/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"


@dataclass
class BoundsSuggestion:
    lower: np.ndarray
    upper: np.ndarray
    mode: str
    reasoning: str
    used_fallback: bool
    raw_response: Optional[str] = None


class LLMBoundsOrchestrator:
    """
    Wraps the LLM call + robust JSON parsing needed to dynamically
    re-shape the BO's search bounds each iteration.
    """

    def __init__(
        self,
        api_key: str,
        cfg: RunConfig,
        config_path: str = "llm_config.json",
        provider: str = "grok",
        timeout_s: float = 30.0,
        raw_log_path: str = "llm_replies_log.txt",
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

        # Initialize native Gemini Client if selected
        if self.provider == "gemini":
            self.gemini_client = genai.Client(api_key=self.api_key)
       

    
    # Public API
    # ------------------------------------------------------------------ #

    def suggest_bounds(
        self,
        last_result: dict,
        gp_uncertainty: float,
        best_score: float,
        current_lower: np.ndarray,
        current_upper: np.ndarray,
        iteration: int,
    ) -> BoundsSuggestion:
        """
        Query the LLM for new search bounds. Never raises — on any failure
        (network, bad JSON, schema violation) it falls back to sane defaults
        and returns used_fallback=True so the caller can log it.
        """
        prompt = self._build_prompt(
            last_result, gp_uncertainty, best_score,
            current_lower, current_upper, iteration,
        )

        try:
            raw_text = self._query_llm(prompt)


            with open(self.raw_log_path, "a", encoding="utf-8") as f:   #––––––––––––––> printing llm replies
                f.write(f"=== ITERATION {iteration} ===\n")
                f.write(raw_text + "\n\n")

                
        except Exception as exc:  # network / auth / provider errors
            return self._fallback(
                current_lower, current_upper,
                reason=f"LLM call failed: {exc}"
            )

        try:
            return self._parse_response(raw_text, current_lower, current_upper)
        except Exception as exc:  # malformed / hallucinated JSON
            return self._fallback(
                current_lower, current_upper,
                reason=f"Could not parse LLM response ({exc})",
                raw_response=raw_text,
            )

    
    # Prompt construction
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self, last_result, gp_uncertainty, best_score,
        current_lower, current_upper, iteration,
    ) -> str:
        k_inf = float(last_result.get("keff", float("nan")))
        ppf = float(last_result.get("ppf", float("nan")))
        delta_k = abs(self.cfg.k_target - k_inf)

        return self.command_template.format(
            iteration=iteration,
            n_bo_iterations=self.cfg.n_bo_iterations,
            n_vars=self.cfg.n_vars,
            k_inf=k_inf,
            k_target=self.cfg.k_target,
            ppf=ppf,
            ppf_target=self.cfg.ppf_target,
            delta_k=delta_k,
            best_score=best_score,
            gp_uncertainty=gp_uncertainty,
            current_lower_preview=np.round(current_lower[:6], 3).tolist(),
            current_upper_preview=np.round(current_upper[:6], 3).tolist(),
        )

    
    # LLM model calls
#--------------------------- #

    
    def _query_llm(self, prompt: str) -> str:

        if self.provider == "anthropic":
            return self._query_anthropic(prompt)
        elif self.provider == "gemini":
            return self._query_gemini(prompt)
        # elif self.provider == "huggingface":
        #     return self._query_huggingface(prompt)   
        elif self.provider == "grok":
            return self._query_grok(prompt)
        elif self.provider == "groq":
            return self._query_groq(prompt)
        elif self.provider == "sarvam":
            return self._query_sarvam(prompt)




# <<<–––––––––––––––––––––– For calling sarwam -––––––––––––––––––––––––––––––>>>>        


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
        
        
        
    
# <<<–––––––––––––––––––––– For calling grog -––––––––––––––––––––––––––––––>>>>        

    def _query_groq(self, prompt: str) -> str:
        """Queries Groq via its OpenAI-compatible chat-completions endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Groq reasoning models perform best when the system and user prompts are combined
        #combined_prompt = self.llm_prompt + "\n\n" + prompt

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
        
        # Optional: If you are using a Groq model that supports JSON mode natively
        # (like llama3-8b-8192 or llama3-70b-8192), you can uncomment the line below 
        # to strictly enforce JSON output at the API level:
        body["response_format"] = {"type": "json_object"}
        
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
       
    

# <<<–––––––––––––––––––––– For calling anthropic -––––––––––––––––––––––––––––––>>>>
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
        resp = requests.post(ANTHROPIC_URL, headers=headers, json=body,
                             timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        return "".join(
            block.get("text", "") for block in data.get("content", [])
            if block.get("type") == "text"
        )

#  <<<–––––––––––––––––––––– For calling grok -––––––––––––––––––––––––––––––>>>>        
    def _query_grok(self, prompt: str) -> str:
        """Queries Grok (xAI) via its OpenAI-compatible chat-completions endpoint."""
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
            "stream": False,  # explicit
        }
        resp = requests.post(
            GROK_URL,
            headers=headers,
            json=body,
            timeout=self.timeout_s
        )
  
        if not resp.ok:
            raise RuntimeError(
                f"HTTP {resp.status_code} Error from x.ai: {resp.text}"
            )
   
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
   
        message = choices[0].get("message", {})
        return message.get("content", "") or ""

    
#  <<<–––––––––––––––––––––– For calling gemini -––––––––––––––––––––––––––––––>>>>    
    def _query_gemini(self, prompt: str) -> str:
        """Queries Gemini using the official SDK and forces a JSON response."""
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




#  <<<–––––––––––––––––––––– For calling huggingface/AtomicGPT -––––––––––––––––––––––––––––––>>>>

    '''
    def _query_huggingface(self, prompt: str) -> str:
         
     HUGGING FACE / ATOMICGPT INTEGRATION NOTE:

     AtomicGPT (e.g., KAERI-MLP/AtomicGPT-gemma2-9B) is registered as a standard
     "Base" text-generation model, not a conversational "Chat" model. Attempting
     to use the `.chat_completion()` endpoint with structured message roles
     results in a 400 Bad Request ('model_not_supported') error.

     NONE OF THE FREE INFERENCE SERVERS HOST ATOMICGPT AT THE MOMENT, and
     there wasn't enough local memory to run it locally either, so this provider path was never completed. The
     "huggingface" option has been fully removed from `provider` validation,
     client init, and `_query_llm` dispatch above.
     
    
    '''




    
    
    
    # Robust JSON parsing / validation
    # -------------------------------- #

    def _extract_json_block(self, raw_text: str) -> dict:
        
        
     ###########################   
        """
        Sometimes the model wraps its JSON in ```json ... ``` even though I said not to in the prompt. 
        so i  striped it off first, json.loads() just blows up since the backticks aren't valid JSON. So: remove the opening fence (with or without the "json" tag),           then the closing one, and clean up any leftover whitespace/newlines each time. - for 'llama' called via 'groq' models -  changes with the models
        """
        text = raw_text.strip()
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    #####################################

        # Locate outermost JSON object
        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if not match:
            raise ValueError("no JSON in the LLM response")
        return json.loads(match.group(1))

   
    def _parse_response(
        self, raw_text: str, current_lower: np.ndarray, current_upper: np.ndarray,
    ) -> BoundsSuggestion:
        payload = self._extract_json_block(raw_text)

        # Safely get mode and reasoning without raising errors -  in case no response
        mode = payload.get("mode", "global")
        reasoning = str(payload.get("reasoning", "no reasonings given  by the llm."))

        if mode == "per_dimension":
            # Use current bounds as a fallback if the LLM forgot to include them
            lower = np.asarray(payload.get("lower_bounds", current_lower), dtype=float)
            upper = np.asarray(payload.get("upper_bounds", current_upper), dtype=float)
            
            if lower.shape != (self.cfg.n_vars,) or upper.shape != (self.cfg.n_vars,):
                raise ValueError(
                    f"per_dimension bounds must have length {self.cfg.n_vars}, "
                    f"got {lower.shape}/{upper.shape}"
                )
        elif mode == "global":
            # Safely get search_radius and center_bias (avoiding KeyError)
            fallback_radius = self.defaults.get("fallback_search_radius", 0.25)
            radius = float(payload.get("search_radius", fallback_radius))
            bias = float(payload.get("center_bias", 0.0))
            
            radius = float(np.clip(radius, self.defaults["min_radius"],
                                    self.defaults["max_radius"]))
            bias = float(np.clip(bias, -1.0, 1.0))

            center = 0.5 * (current_lower + current_upper)
            # when bias > 0 -> moves the center toward 1 (high enrich),
            # bias < 0 -> moves it toward 0 (low enrich)
            center = center + bias * 0.5 * (1.0 - np.abs(2.0 * center - 1.0))
            lower = center - radius
            upper = center + radius
        else:
            # If the LLM hallucinated a weird mode name, force global
            mode = "global"
            radius = self.defaults.get("fallback_search_radius", 0.25)
            center = 0.5 * (current_lower + current_upper)
            lower = center - radius
            upper = center + radius

        '''
        Hard safety clip — never allow the LLM to push the search outside the physically valid [0, 1] normalized design space, 
        and never allow an inverted (lower > upper) interval.
        '''
        lower = np.clip(lower, self.cfg.lower, self.cfg.upper)
        upper = np.clip(upper, self.cfg.lower, self.cfg.upper)
        inverted = lower > upper
        if np.any(inverted):
            lower[inverted], upper[inverted] = upper[inverted], lower[inverted]
            
        # Guarantee a minimum usable width so EI never degenerates on a point
        too_narrow = (upper - lower) < 1e-3
        upper[too_narrow] = np.minimum(1.0, lower[too_narrow] + 1e-3)

        return BoundsSuggestion(
            lower=lower, upper=upper, mode=mode, reasoning=reasoning,
            used_fallback=False, raw_response=raw_text,
        )
    
    
    
    def _fallback(
        self, current_lower, current_upper, reason: str,
        raw_response: Optional[str] = None,
    ) -> BoundsSuggestion:
        """
        Safety  - On failure (any), spread slightly around the current center using the
        fallback radius rather than crashing the optimisation loop.
        """
        radius = self.defaults["fallback_search_radius"]
        center = 0.5 * (current_lower + current_upper)
        lower = np.clip(center - radius, self.cfg.lower, self.cfg.upper)
        upper = np.clip(center + radius, self.cfg.lower, self.cfg.upper)
        return BoundsSuggestion(
            lower=lower, upper=upper, mode=self.defaults["fallback_mode"],
            reasoning=f"[fallback] {reason}", used_fallback=True,
            raw_response=raw_response,
        )