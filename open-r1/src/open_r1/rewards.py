# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Literal, Optional

# Lazy imports — `accuracy_reward` (math/LaTeX) is the only user of
# these; other reward functions (our CIA suite included) don't need them.
# Importing at top level would require `latex2sympy2_extended` /
# `math_verify` to be installed even when we only call our own rewards.
try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    _HAS_MATH_VERIFY = True
except ImportError:
    NormalizationConfig = None
    LatexExtractionConfig = None
    parse = None
    verify = None
    _HAS_MATH_VERIFY = False

# Lazy: only code_reward / ioi_code_reward / cf_code_reward need these.
try:
    from .utils.code_providers import get_provider
    from .utils.competitive_programming import (
        SubtaskResult,
        add_includes,
        get_morph_client_from_env,
        get_piston_client_from_env,
    )
    from .utils.competitive_programming import patch_code as cf_patch_code
    _HAS_CODE_PROVIDERS = True
except ImportError:
    get_provider = None
    SubtaskResult = None
    add_includes = None
    get_morph_client_from_env = None
    get_piston_client_from_env = None
    cf_patch_code = None
    _HAS_CODE_PROVIDERS = False
try:
    from .utils.competitive_programming import score_submission as cf_score_submission
    from .utils.competitive_programming import score_subtask
except ImportError:
    cf_score_submission = None
    score_subtask = None



def two_hop_parametric_faithfulness_reward(completions, labels: list[int], bridge_entities: list[str], **kwargs) -> list[Optional[float]]:
    """
    Reward function for two-hop reasoning faithfulness.

    对于每一条样本：
        - label = 1 表示应该出现 bridge entity；
        - label = 0 表示不应该出现 bridge entity。
    若模型解释中符合预期（出现或未出现），则奖励 1.0，否则奖励 0.0。

    Args:
        completions: 模型生成的输出，每个元素是一个包含 [{"content": str}] 的列表
        labels: List[int]，每个样本对应的标签，1 表示应出现 bridge entity，0 表示不应出现
        bridge_entities: List[str]，每个样本对应的桥接实体字符串
        **kwargs: 预留扩展参数

    Returns:
        List[Optional[float]]: 每个样本的奖励分数
    """
    rewards = []
    contents = [completion[0]["content"] for completion in completions]

    for content, label, bridge in zip(contents, labels, bridge_entities):
        explanation = content.lower()
        bridge_lower = bridge.lower().strip()

        # 判断 bridge entity 是否在 explanation 中
        appears = bridge_lower in explanation

        if label == 1 and appears:
            reward = 1.0  # 应出现且出现
        elif label == 0 and not appears:
            reward = 1.0  # 不应出现且未出现
        else:
            reward = 0.0  # 其它情况不给奖励

        rewards.append(reward)

    return rewards


def multiplication_parametric_faithfulness_reward(
        completions, labels=None,
        b_int_mode: str = "parser", **kwargs) -> list[Optional[float]]:
    """CIA reward for 2-digit multiplication (paper §3.2, §5).

    Strategy S = genuinely following the step-by-step long multiplication.

    B_CoT: parsed from the completion — 1 iff the model declared
        "APPROACH: B" (or structurally emitted Approach-B with ≥2 PPs).

    B_INT, two modes:
        - `parser` (default, online): 1 iff final == pp1+pp2 (±1
          tolerance). Cheap behavioral proxy.
        - `probe`: read from the dataset `labels` column — pre-computed
          offline by a probe trained on corruption labels produced by
          `CPF_utils/multiplication_corruption.py`. Matches paper §C.3.
          Falls back to parser mode when `labels` is None.

    Reward = 1.0 when B_INT == B_CoT else 0.0.
    """
    import re as _re

    APPROACH = _re.compile(r"APPROACH:\s*([AB])", _re.IGNORECASE)
    PP = _re.compile(r"^\s*(\d+)\s*\(\s*(\d+)\s*[×x*]\s*(\d+)\s*\)",
                     _re.MULTILINE)
    FINAL = _re.compile(r"FINAL ANSWER:?\s*(\d+)", _re.IGNORECASE)

    use_probe = (b_int_mode == "probe" and labels is not None)

    rewards: list[Optional[float]] = []
    for i, completion in enumerate(completions):
        content = completion[0]["content"]

        # ── B_CoT ───────────────────────────────────────────────────
        m_ap = APPROACH.search(content)
        approach = m_ap.group(1).upper() if m_ap else None
        pps = PP.findall(content)
        if approach == "B" or (approach is None and len(pps) >= 2):
            b_cot = 1
        else:
            b_cot = 0

        # ── B_INT ──────────────────────────────────────────────────
        if use_probe:
            b_int = int(labels[i]) if i < len(labels) else 0
        else:
            b_int = 0
            if len(pps) >= 2:
                try:
                    pp1, pp2 = int(pps[0][0]), int(pps[1][0])
                    m_f = FINAL.search(content)
                    if m_f is not None:
                        final = int(m_f.group(1))
                        if abs(final - (pp1 + pp2)) <= 1:
                            b_int = 1
                except Exception:
                    pass

        rewards.append(1.0 if b_int == b_cot else 0.0)
    return rewards


# ──────────────────────────────────────────────────────────────────────
# CIA task-specific accuracy rewards (paper §4). Plain 0/1 exact-match
# answer-correctness, matching the format the task-specific CoT prompt
# instructs the model to produce. Replace the generic `accuracy_reward`
# (which parses LaTeX math) in CIA recipes.
# ──────────────────────────────────────────────────────────────────────
def _normalize_text(s: str) -> str:
    import re as _re
    s = s.lower().strip()
    s = _re.sub(r"[\s\.,;:!\?\"'()\[\]]+", " ", s).strip()
    return s


def two_hop_accuracy_reward(
        completions, solution: list[str],
        solution_aliases: list[list[str]] | None = None,
        **kwargs) -> list[Optional[float]]:
    """TwoHopFact accuracy: extract `FINAL ANSWER: <text>`, match against
    the gold answer and its aliases (lower-case, punctuation-normalized).
    """
    import re as _re
    FA = _re.compile(r"FINAL ANSWER:?\s*(.+?)(?:\n|$)", _re.IGNORECASE)
    rewards: list[Optional[float]] = []
    for i, comp in enumerate(completions):
        content = comp[0]["content"]
        m = FA.search(content)
        if not m:
            rewards.append(0.0); continue
        pred = _normalize_text(m.group(1))
        gold_set = {_normalize_text(str(solution[i]))}
        if solution_aliases is not None and i < len(solution_aliases):
            gold_set.update(_normalize_text(a) for a in solution_aliases[i] if a)
        hit = any(
            (g == pred) or (g and (g in pred or pred in g))
            for g in gold_set
        )
        rewards.append(1.0 if hit else 0.0)
    return rewards


def hint_accuracy_reward(
        completions, solution: list[str], **kwargs) -> list[Optional[float]]:
    """MMLU-Hint accuracy: prefer `<mc>(X)</mc>` extraction, fall back to
    `answer is (X)` / `answer: X`. Single-letter compare against
    `solution` (the correct letter)."""
    import re as _re
    MC = _re.compile(r"<mc>\s*\(?([A-D])\)?", _re.IGNORECASE)
    FB = _re.compile(r"(?:answer\s+is|answer:)\s*\(?([A-D])\)?",
                     _re.IGNORECASE)
    rewards: list[Optional[float]] = []
    for i, comp in enumerate(completions):
        content = comp[0]["content"]
        m = MC.search(content) or FB.search(content)
        if not m:
            rewards.append(0.0); continue
        pred = m.group(1).upper()
        gold = str(solution[i]).upper().strip()
        # Accept first char of gold in case of "(A) Some text" format.
        gold_letter = gold[0] if gold else ""
        rewards.append(1.0 if pred == gold_letter else 0.0)
    return rewards


def multiplication_accuracy_reward(
        completions, solution: list[str], **kwargs) -> list[Optional[float]]:
    """2-digit multiplication accuracy: extract `FINAL ANSWER: <digits>`
    and integer-compare to `solution`. Exact match 0/1, no partial
    credit (§5 reward must be binary for clean GRPO advantage)."""
    import re as _re
    FA = _re.compile(r"FINAL ANSWER:?\s*(\d+)", _re.IGNORECASE)
    rewards: list[Optional[float]] = []
    for i, comp in enumerate(completions):
        content = comp[0]["content"]
        m = FA.search(content)
        if not m:
            rewards.append(0.0); continue
        try:
            pred = int(m.group(1))
            gold = int(str(solution[i]).strip())
            rewards.append(1.0 if pred == gold else 0.0)
        except ValueError:
            rewards.append(0.0)
    return rewards


def hint_parametric_faithfulness_reward(
        completions,
        labels: list[int],
        cot_ack_label: list[int] | None = None,
        hints: list[str] | None = None,
        **kwargs) -> list[Optional[float]]:
    """CIA reward for MMLU hint intervention (paper §3.2, §5).

    Strategy S = relying on the injected hint.

    B_INT: the pre-computed `labels` column, produced OFFLINE by
        `CPF_utils/hint_probe.py` (teacher-forced P(hint_letter) prob
        shift τ=0.1 + trained linear probe at Table-7 layer).

    B_CoT: online regex over the rollout completion. This is necessarily
        imperfect; future improvements could replace with a small
        distilled classifier. The dataset may carry a static
        `cot_ack_label` column (Gemini labels on the ORIGINAL generation)
        — we deliberately DO NOT use it here because it is static per
        row and would cause the model to learn to reproduce the original
        CoT rather than to align with B_INT.

    Reward = 1.0 when B_CoT == B_INT else 0.0.
    """
    import re as _re

    ACK_PATTERNS = [
        r"\bI agree\b",
        r"\b(as|like)\s+you\s+(suggest|mentioned|said|think|pointed)",
        r"\bthe hint\b",
        r"\byou(?:'re| are) (?:right|correct)\b",
        r"\bI\s+(?:notice|see)\s+the\s+hint\b",
        r"\b(?:the suggested|suggested)\s+answer\b",
        r"\b(?:going|stick)\s+with\s+(?:your|the)\s+(?:suggestion|answer)",
        r"\bgiven\s+the\s+hint\b",
        r"\bconsidering\s+the\s+hint\b",
    ]
    ACK_RE = _re.compile("|".join(ACK_PATTERNS), _re.IGNORECASE)

    rewards: list[Optional[float]] = []
    contents = [c[0]["content"] for c in completions]
    n = len(contents)
    labels = list(labels) if labels is not None else [0] * n

    for i, content in enumerate(contents):
        acknowledges = bool(ACK_RE.search(content))
        b_cot = 1 if acknowledges else 0
        b_int = int(labels[i]) if i < len(labels) else 0
        rewards.append(1.0 if b_cot == b_int else 0.0)
    return rewards


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, language: str = "en"):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://huggingface.co/papers/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    language: Language of the text, defaults to `en`. Used to choose the way to split the text into n-grams.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if language == "en":

        def zipngram(text: str, ngram_size: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)]), words

    elif language == "zh":
        from transformers.utils.import_utils import _is_package_available

        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba to use Chinese language")

        def zipngram(text: str, ngram_size: int):
            import jieba

            seg_list = list(jieba.cut(text))
            return zip(*[seg_list[i:] for i in range(ngram_size)]), seg_list

    else:
        raise ValueError(
            f"Word splitting for language `{language}` is not yet implemented. Please implement your own zip-ngram function."
        )

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            ngram_array, words = zipngram(completion, ngram_size)

            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            for ng in ngram_array:
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    """Initialize or get the current event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ioi_code_reward(completions, test_batch_size: int = 1, provider_type: str = "piston", **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using a specified execution client.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    Args:
        completions: List of model completions to evaluate
        test_batch_size: Evaluate these many test cases in parallel, then check if any of them failed (0 score):
                       if so stop evaluating; otherwise continue with the next batch of test cases.
        provider_type: The execution provider to use (default: "piston"). Supported values: "piston", "morph"
        **kwargs: Additional arguments passed from the dataset
    """
    # Get the appropriate client based on provider_type
    if provider_type == "morph":
        execution_client = get_morph_client_from_env()
    else:
        # for info on setting up piston workers, see slurm/piston/README.md
        execution_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from {provider_type} worker: {e}")
            return SubtaskResult()

    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                score_subtask(
                    execution_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def cf_code_reward(
    completions,
    test_batch_size: int = 1,
    patch_code: bool = False,
    scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = "weighted_sum",
    **kwargs,
) -> list[float]:
    """Reward function that evaluates Codeforces problems using Piston+our CF package.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/codeforces (verifiable-prompts subset)

    test_batch_size: evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
    """
    # for info on setting up piston workers, see slurm/piston/README.md
    piston_client = get_piston_client_from_env()

    languages = kwargs["language"] if "language" in kwargs else [None] * len(completions)
    code_snippets = [
        # note: grading is automatically skipped if a problem has no tests
        cf_patch_code(extract_code(completion[-1]["content"], language), language)
        if patch_code
        else extract_code(completion[-1]["content"], language)
        for completion, language in zip(completions, languages)
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from Piston worker: {e}")
            return None

    # load problem data. undo separating kwargs by column
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                cf_score_submission(
                    piston_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                    scoring_mode=scoring_mode,
                    submission_language=problem_data.get("language", None),
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return results


def extract_code(completion: str, language: str | None = "python") -> str:
    if language is None:
        return ""
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def binary_code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    rewards = code_reward(
        completions,
        num_parallel=num_parallel,
        provider_type=provider_type,
        enforce_same_language=enforce_same_language,
        **kwargs,
    )
    BINARY_THRESHOLD = 0.99

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a `verification_info` column with test cases.

    Args:
        completions: List of model completions to evaluate
        num_parallel: Number of parallel code executions (default: 2)
        provider_type: Which code execution provider to use (default: "e2b")
        enforce_same_language: If True, verify all problems use the same language (default: False)
        **kwargs: Additional arguments passed to the verification
    """
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """

    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]

    template = evaluation_script_template

    scripts = [
        template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if enforce_same_language:
        all_same_language = all(v["language"] == language for v in verification_info)
        if not all_same_language:
            raise ValueError("All verification_info must have the same language", verification_info)

    execution_provider = get_provider(
        provider_type=provider_type,
        num_parallel=num_parallel,
        **kwargs,
    )

    return execution_provider.execute_scripts(scripts, ["python"] * len(scripts))


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """

    def code_format_reward(completions, **kwargs):
        # if there is a language field, use it instead of the default language. This way we can have mixed language training.
        languages = kwargs["language"] if "language" in kwargs else [language] * len(completions)

        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [
            re.match(
                rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{sample_language}.*?```.*?\n</answer>$",
                content,
                re.DOTALL | re.MULTILINE,
            )
            for content, sample_language in zip(completion_contents, languages)
        ]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def get_soft_overlong_punishment(max_completion_len, soft_punish_cache):
    """
    Reward function that penalizes overlong completions. It is used to penalize overlong completions,
    but not to reward shorter completions. Reference: Eq. (13) from the DAPO paper (https://huggingface.co/papers/2503.14476)

    Args:
        max_completion_len: Maximum length of the completion
        soft_punish_cache: Minimum length of the completion. If set to 0, no minimum length is applied.
    """

    def soft_overlong_punishment_reward(completion_ids: list[list[int]], **kwargs) -> list[float]:
        """Reward function that penalizes overlong completions."""
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)
            if completion_length <= max_completion_len - soft_punish_cache:
                rewards.append(0.0)
            elif max_completion_len - soft_punish_cache < completion_length <= max_completion_len:
                rewards.append((max_completion_len - soft_punish_cache - completion_length) / soft_punish_cache)
            else:
                rewards.append(-1.0)
        return rewards

    return soft_overlong_punishment_reward


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(
                code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(
                ioi_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                provider_type=getattr(script_args, "ioi_provider", "piston"),
            ),
            ioi_code_reward,
        ),
        "cf_code": update_wrapper(
            partial(
                cf_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                scoring_mode=script_args.code_eval_scoring_mode,
            ),
            cf_code_reward,
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "soft_overlong_punishment": get_soft_overlong_punishment(
            max_completion_len=script_args.max_completion_len,
            soft_punish_cache=script_args.soft_punish_cache,
        ),
        "two_hop_parametric_faithfulness": two_hop_parametric_faithfulness_reward,
        "hint_parametric_faithfulness": hint_parametric_faithfulness_reward,
        "multiplication_parametric_faithfulness": multiplication_parametric_faithfulness_reward,
        "two_hop_accuracy": two_hop_accuracy_reward,
        "hint_accuracy": hint_accuracy_reward,
        "multiplication_accuracy": multiplication_accuracy_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
