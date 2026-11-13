"""Deterministic synthetic document and needle-in-a-haystack retrieval dataset generator."""

import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class RetrievalItem:
    """A single retrieval test instance."""

    item_id: str
    target_fact: str
    question: str
    ground_truth_answer: str
    context_tokens_target: int
    actual_prompt_tokens: int
    fact_position_fraction: float  # 0.1 (beginning), 0.5 (middle), 0.9 (end)
    position_bucket: str  # "beginning", "middle", "end"
    full_prompt: str
    split: str  # "calibration" or "held_out"


# Standard background filler sentences (deterministic corpus)
FILLER_SENTENCES = [
    "The atmospheric pressure at sea level remains approximately 1013.25 hectopascals under standard conditions.",
    "Silicon carbide semiconductors demonstrate high breakdown voltage and superior thermal conductivity.",
    "Distributed consensus algorithms ensure state machine replication across unreliable networks.",
    "Oceanic thermohaline circulation is driven by global density gradients created by surface heat and freshwater fluxes.",
    "Cache replacement policies such as least recently used minimize memory latency in multi-tiered architectures.",
    "Photosynthetic electron transport in plants converts light energy into chemical energy through coupled redox reactions.",
    "The fast Fourier transform computes the discrete Fourier transform in asymptotic time proportional to N log N.",
    "Geostationary satellites maintain an orbital period equal to the Earth's rotational period at 35786 kilometers.",
    "Asynchronous I/O event loops allow single-threaded systems to achieve high concurrent connection density.",
    "In quantum mechanics, energy quantization arises naturally from boundary conditions on the wave function.",
    "Sedimentary rock stratification reflects historical cycles of erosion, deposition, and lithification over geological epochs.",
    "Compiler instruction scheduling optimizes execution pipelining while preserving data and control dependencies.",
]

CALIBRATION_FACTS = [
    ("The secret operational frequency of the beacon is 433.92 megahertz.", "What is the secret operational frequency of the beacon?", "433.92 megahertz"),
    ("The emergency override code for sector seven is Delta-99-Gamma.", "What is the emergency override code for sector seven?", "Delta-99-Gamma"),
    ("The subterranean laboratory is located at coordinates latitude 45.12 and longitude 12.84.", "What are the coordinates of the subterranean laboratory?", "latitude 45.12 and longitude 12.84"),
    ("The primary coolant used in reactor core B is liquid fluorochemical FC-770.", "What is the primary coolant used in reactor core B?", "liquid fluorochemical FC-770"),
    ("The designated archive password for project Chronos is Obsidian-Vortex-2024.", "What is the designated archive password for project Chronos?", "Obsidian-Vortex-2024"),
    ("The acoustic signature frequency of vessel Triton is 142.6 hertz.", "What is the acoustic signature frequency of vessel Triton?", "142.6 hertz"),
]

HELDOUT_FACTS = [
    ("The security access key for vault eleven is Cipher-Theta-884.", "What is the security access key for vault eleven?", "Cipher-Theta-884"),
    ("The hidden calibration offset of the sensor array is exactly 3.1415 millivolts.", "What is the hidden calibration offset of the sensor array?", "3.1415 millivolts"),
    ("The rendezvous coordinates for extraction team Alpha are grid sector Omega-42.", "What are the rendezvous coordinates for extraction team Alpha?", "grid sector Omega-42"),
    ("The specialized alloy composition for turbine blades is nickel-chromium alloy Nimonic-90.", "What is the specialized alloy composition for turbine blades?", "nickel-chromium alloy Nimonic-90"),
    ("The encryption initialization vector for message packet zero is 0x7F2A9B1C.", "What is the encryption initialization vector for message packet zero?", "0x7F2A9B1C"),
    ("The measured stellar parallax of the target star system is 12.45 milliarcseconds.", "What is the measured stellar parallax of the target star system?", "12.45 milliarcseconds"),
    ("The thermal degradation threshold of sealant compound K-4 is 482.5 degrees Celsius.", "What is the thermal degradation threshold of sealant compound K-4?", "482.5 degrees Celsius"),
    ("The orbital inclination of reconnaissance platform Zephyr is 64.8 degrees.", "What is the orbital inclination of reconnaissance platform Zephyr?", "64.8 degrees"),
    ("The cryptographic authentication hash for node Echo is SHA256-9F81E3B7.", "What is the cryptographic authentication hash for node Echo?", "SHA256-9F81E3B7"),
    ("The specific impulse of propellant mixture Hydrazine-7 is 312 seconds.", "What is the specific impulse of propellant mixture Hydrazine-7?", "312 seconds"),
    ("The quantum coherence dephasing time for qubit cluster four is 84.2 microseconds.", "What is the quantum coherence dephasing time for qubit cluster four?", "84.2 microseconds"),
    ("The resonant acoustic frequency of the containment chamber is 528.4 hertz.", "What is the resonant acoustic frequency of the containment chamber?", "528.4 hertz"),
    ("The maximum payload capacity of cargo module Atlas is 14250 kilograms.", "What is the maximum payload capacity of cargo module Atlas?", "14250 kilograms"),
    ("The magnetic flux density at the focal center is 2.85 teslas.", "What is the magnetic flux density at the focal center?", "2.85 teslas"),
    ("The optical absorption coefficient of protective filter V-9 is 99.82 percent.", "What is the optical absorption coefficient of protective filter V-9?", "99.82 percent"),
    ("The hydraulic line pressure in landing gear assembly C is 3050 psi.", "What is the hydraulic line pressure in landing gear assembly C?", "3050 psi"),
    ("The telemetry baud rate for deep-space probe Horizon is 115200 symbols per second.", "What is the telemetry baud rate for deep-space probe Horizon?", "115200 symbols per second"),
    ("The critical superconducting transition temperature of wire batch N-12 is 9.3 kelvins.", "What is the critical superconducting transition temperature of wire batch N-12?", "9.3 kelvins"),
    ("The biological half-life of tracing isotope Technetium-99m in sample Q is 6.01 hours.", "What is the biological half-life of tracing isotope Technetium-99m in sample Q?", "6.01 hours"),
    ("The safe working depth of research submersible Abyss-3 is 4200 meters.", "What is the safe working depth of research submersible Abyss-3?", "4200 meters"),
]


class RetrievalDatasetGenerator:
    """Generates synthetic long-context documents with needle facts embedded at controlled depths."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def build_item(
        self,
        item_id: str,
        fact_tuple: Tuple[str, str, str],
        target_tokens: int,
        position_fraction: float,
        split: str = "calibration",
    ) -> RetrievalItem:
        statement, question, answer = fact_tuple

        if position_fraction <= 0.33:
            pos_bucket = "beginning"
        elif position_fraction <= 0.66:
            pos_bucket = "middle"
        else:
            pos_bucket = "end"

        # Construct background filler to reach target token count
        # Approximate 1 sentence ~ 16 tokens
        target_filler_tokens = max(100, target_tokens - 100)
        rng = random.Random(hash(f"{item_id}_{target_tokens}_{position_fraction}") & 0xFFFFFFFF)

        filler_blocks = []
        accum_tokens = 0
        while accum_tokens < target_filler_tokens:
            sent = rng.choice(FILLER_SENTENCES)
            filler_blocks.append(sent)
            accum_tokens += len(sent.split()) * 1.3  # rough estimate

        # Insert needle at specified position
        insert_idx = int(len(filler_blocks) * position_fraction)
        insert_idx = min(len(filler_blocks), max(0, insert_idx))

        prefix = " ".join(filler_blocks[:insert_idx])
        suffix = " ".join(filler_blocks[insert_idx:])

        full_doc = f"{prefix}\n\n[CRITICAL FACT]: {statement}\n\n{suffix}"
        prompt = (
            f"<|im_start|>system\nYou are a precise factual retrieval assistant. Answer the question using ONLY the provided document. "
            f"Be concise and extract the exact fact.<|im_end|>\n"
            f"<|im_start|>user\nDOCUMENT:\n{full_doc}\n\nQUESTION: {question}\nProvide the exact answer directly without explanation.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        actual_tokens = (
            len(self.tokenizer.encode(prompt))
            if self.tokenizer is not None
            else int(len(prompt.split()) * 1.3)
        )

        # If actual tokens diverge significantly from target and tokenizer is available, adjust
        if self.tokenizer is not None and abs(actual_tokens - target_tokens) > 100:
            token_ids = self.tokenizer.encode(prompt)
            if len(token_ids) > target_tokens:
                # Trim filler from suffix or prefix while retaining system, fact, and user question
                # To be exact, re-generate with calibrated sentence count
                ratio = target_tokens / float(actual_tokens)
                adjusted_num_sents = max(5, int(len(filler_blocks) * ratio))
                adj_prefix = " ".join(filler_blocks[: int(adjusted_num_sents * position_fraction)])
                adj_suffix = " ".join(filler_blocks[int(adjusted_num_sents * position_fraction) : adjusted_num_sents])
                full_doc = f"{adj_prefix}\n\n[CRITICAL FACT]: {statement}\n\n{adj_suffix}"
                prompt = (
                    f"<|im_start|>system\nYou are a precise factual retrieval assistant. Answer the question using ONLY the provided document. "
                    f"Be concise and extract the exact fact.<|im_end|>\n"
                    f"<|im_start|>user\nDOCUMENT:\n{full_doc}\n\nQUESTION: {question}\nProvide the exact answer directly without explanation.<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                actual_tokens = len(self.tokenizer.encode(prompt))

        return RetrievalItem(
            item_id=item_id,
            target_fact=statement,
            question=question,
            ground_truth_answer=answer,
            context_tokens_target=target_tokens,
            actual_prompt_tokens=actual_tokens,
            fact_position_fraction=position_fraction,
            position_bucket=pos_bucket,
            full_prompt=prompt,
            split=split,
        )

    def generate_suite(
        self,
        context_lengths: List[int],
        split: str = "calibration",
        positions: Optional[List[float]] = None,
    ) -> List[RetrievalItem]:
        positions = positions or [0.1, 0.5, 0.9]  # beginning, middle, end
        facts_pool = CALIBRATION_FACTS if split == "calibration" else HELDOUT_FACTS

        items = []
        for ctx_len in context_lengths:
            for p_idx, pos in enumerate(positions):
                fact_idx = (p_idx + context_lengths.index(ctx_len)) % len(facts_pool)
                fact_tuple = facts_pool[fact_idx]
                item_id = f"{split}_ctx{ctx_len}_pos{int(pos*100)}_{fact_idx}"
                item = self.build_item(
                    item_id=item_id,
                    fact_tuple=fact_tuple,
                    target_tokens=ctx_len,
                    position_fraction=pos,
                    split=split,
                )
                items.append(item)
        return items
