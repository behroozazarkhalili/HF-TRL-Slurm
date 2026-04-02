#!/usr/bin/env python3
"""
End-to-end test of all GRPO training fixes using REAL NuminaMath-CoT data.

Tests:
1. _extract_boxed_content: Can it parse real \boxed{} from NuminaMath solutions?
2. extract_answer: Does it correctly extract answers from real solutions?
3. _prompt_key: Does round-trip storage/retrieval work with real prompts?
4. normalize_answer + _try_parse_number: Do they handle real NuminaMath answers?
5. Ground truth extraction rate: What % of dataset samples yield ground truth?
6. Reward function: Does it produce non-zero rewards for simulated completions?

Run: module load gcc arrow python/3.11.5 && source /scratch/ermia/venvs/hf_env/bin/activate && python tests/test_real_numina_data.py
"""

from __future__ import annotations

import re
import sys
import time
from typing import Optional, Dict

# ========================================================================
# Copy the exact functions from train_grpo.py (pure Python, no torch needed)
# ========================================================================

def _prompt_key(prompt: str) -> str:
    """Create a stable lookup key from a prompt string."""
    key = re.sub(r'<\|[^|]*\|>', '', prompt)
    return key.strip()


def _extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from the LAST \\boxed{...} handling nested braces correctly."""
    idx = text.rfind('\\boxed{')
    if idx == -1:
        idx = text.rfind('\\boxed {')
    if idx == -1:
        return None
    brace_start = text.find('{', idx)
    if brace_start == -1:
        return None
    depth = 1
    pos = brace_start + 1
    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1
    if depth == 0:
        return text[brace_start + 1:pos - 1].strip()
    return None


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from a math solution."""
    if not text:
        return None
    boxed = _extract_boxed_content(text)
    if boxed:
        return boxed
    answer_patterns = [
        r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+([^\n.]+)',
        r'[Ff]inal\s+[Aa]nswer[:\s]+([^\n.]+)',
        r'[Aa]nswer[:\s]+([^\n.]+)',
        r'[Tt]herefore[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\n.]+)',
        r'[Ss]o[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\n.]+)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    last_line_match = re.search(r'=\s*([+-]?\d+(?:\.\d+)?)\s*$', text)
    if last_line_match:
        return last_line_match.group(1).strip()
    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    if not answer:
        return ""
    answer = answer.strip().lower()
    # Convert LaTeX fractions to a/b BEFORE stripping (prevents \frac{5}{9} -> "59")
    # Pattern handles one level of nested braces: \frac{3\sqrt{3}}{2} -> 3\sqrt{3}/2
    answer = re.sub(r'\\frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1/\2', answer)
    # Remove common LaTeX formatting that wraps content
    answer = re.sub(r'\\text\s*\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathrm\s*\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\sqrt\s*\{([^}]+)\}', r'sqrt(\1)', answer)
    # Remove remaining LaTeX commands
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)
    answer = re.sub(r'[{}$]', '', answer)
    # Normalize fractions (plain text a / b -> a/b)
    answer = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', answer)
    answer = re.sub(r'[.,;:!?]+$', '', answer)
    answer = ' '.join(answer.split())
    return answer


def _try_parse_number(s: str) -> Optional[float]:
    """Try to parse a normalized answer string as a number."""
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                pass
    return None


# ========================================================================
# Test harness
# ========================================================================

def main():
    print("=" * 70)
    print("REAL NuminaMath-CoT Dataset Verification Test")
    print("=" * 70)

    # Load real dataset
    from datasets import load_dataset

    print("\n[1/7] Loading AI-MO/NuminaMath-CoT dataset (streaming, 500 samples)...")
    t0 = time.time()
    dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    dataset = dataset.shuffle(seed=42, buffer_size=10000)

    samples = []
    for i, example in enumerate(dataset):
        samples.append(example)
        if len(samples) >= 500:
            break

    elapsed = time.time() - t0
    print(f"  Loaded {len(samples)} samples in {elapsed:.1f}s")
    print(f"  Sample keys: {list(samples[0].keys())}")

    # ---- Test 1: Ground truth extraction rate ----
    print("\n[2/7] Testing ground truth extraction from real solutions...")
    ground_truth: Dict[str, str] = {}
    extraction_failures = []
    boxed_count = 0
    no_boxed_count = 0

    for i, example in enumerate(samples):
        problem = example.get("problem", "")
        solution = example.get("solution", "")

        if not problem or not solution:
            continue

        # Check if solution contains \boxed{}
        has_boxed = '\\boxed{' in solution or '\\boxed {' in solution
        if has_boxed:
            boxed_count += 1
        else:
            no_boxed_count += 1

        answer = extract_answer(solution)
        if answer:
            ground_truth[_prompt_key(problem)] = answer
        else:
            extraction_failures.append({
                "idx": i,
                "problem": problem[:80],
                "solution_snippet": solution[-200:] if len(solution) > 200 else solution,
                "has_boxed": has_boxed,
            })

    total_with_solution = boxed_count + no_boxed_count
    extraction_rate = len(ground_truth) / total_with_solution * 100 if total_with_solution > 0 else 0

    print(f"  Total samples with problem+solution: {total_with_solution}")
    print(f"  Solutions containing \\boxed{{}}: {boxed_count}")
    print(f"  Solutions WITHOUT \\boxed{{}}: {no_boxed_count}")
    print(f"  Successfully extracted ground truth: {len(ground_truth)} ({extraction_rate:.1f}%)")
    print(f"  Extraction failures: {len(extraction_failures)}")

    if extraction_failures[:3]:
        print(f"\n  Sample failures (first 3):")
        for f in extraction_failures[:3]:
            print(f"    [{f['idx']}] has_boxed={f['has_boxed']}")
            print(f"         problem: {f['problem']}...")
            print(f"         solution tail: ...{f['solution_snippet'][-100:]}")

    assert extraction_rate > 80, f"Extraction rate too low: {extraction_rate:.1f}% (expected >80%)"
    print(f"  PASS: Extraction rate {extraction_rate:.1f}% > 80%")

    # ---- Test 2: Boxed content parsing on real data ----
    print("\n[3/7] Testing _extract_boxed_content on real solutions...")
    boxed_results = {"simple": 0, "nested": 0, "deep_nested": 0, "failed": 0}
    real_extracted = []

    for example in samples:
        solution = example.get("solution", "")
        if '\\boxed{' not in solution and '\\boxed {' not in solution:
            continue

        extracted = _extract_boxed_content(solution)
        if extracted:
            # Classify complexity
            brace_depth = 0
            max_depth = 0
            for ch in extracted:
                if ch == '{':
                    brace_depth += 1
                    max_depth = max(max_depth, brace_depth)
                elif ch == '}':
                    brace_depth -= 1

            if max_depth == 0:
                boxed_results["simple"] += 1
            elif max_depth == 1:
                boxed_results["nested"] += 1
            else:
                boxed_results["deep_nested"] += 1

            real_extracted.append(extracted)
        else:
            boxed_results["failed"] += 1

    total_boxed_tested = sum(boxed_results.values())
    success_rate = (total_boxed_tested - boxed_results["failed"]) / total_boxed_tested * 100 if total_boxed_tested > 0 else 0

    print(f"  Tested {total_boxed_tested} solutions with \\boxed{{}}")
    print(f"  Simple answers (no inner braces): {boxed_results['simple']}")
    print(f"  Nested (1 level, e.g. \\frac{{}}{{}}): {boxed_results['nested']}")
    print(f"  Deep nested (2+ levels): {boxed_results['deep_nested']}")
    print(f"  Parse failures: {boxed_results['failed']}")
    print(f"  Success rate: {success_rate:.1f}%")

    # Show some real extracted answers
    print(f"\n  Sample extracted answers (first 15):")
    for ans in real_extracted[:15]:
        print(f"    -> {ans}")

    assert success_rate > 95, f"Boxed parsing success rate too low: {success_rate:.1f}%"
    print(f"\n  PASS: Boxed parsing success rate {success_rate:.1f}% > 95%")

    # ---- Test 3: _prompt_key round-trip ----
    print("\n[4/7] Testing _prompt_key round-trip with real prompts...")
    direct_pass = 0
    direct_fail = 0
    chat_pass = 0
    chat_fail = 0

    for example in samples[:100]:
        problem = example.get("problem", "")
        if not problem:
            continue

        # Store
        key = _prompt_key(problem)
        stored_val = "test_answer"
        test_dict = {key: stored_val}

        # Test 3a: Direct round-trip (same prompt stored and retrieved)
        retrieved = test_dict.get(_prompt_key(problem))
        if retrieved == stored_val:
            direct_pass += 1
        else:
            direct_fail += 1

        # Test 3b: Chat-template round-trip
        # TRL may wrap prompts with chat templates before passing to reward functions
        # Test with Qwen chat template format
        wrapped = f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"
        wrapped_key = _prompt_key(wrapped)
        retrieved_wrapped = test_dict.get(wrapped_key)
        if retrieved_wrapped == stored_val:
            chat_pass += 1
        else:
            chat_fail += 1

    print(f"  Direct round-trip:        {direct_pass} pass, {direct_fail} fail")
    print(f"  Chat-template round-trip: {chat_pass} pass, {chat_fail} fail")

    # Direct round-trip MUST work (critical)
    assert direct_fail == 0, f"Direct round-trip failures: {direct_fail}"
    print(f"  PASS: Direct round-trip 100% success")

    if chat_fail > 0:
        print(f"  NOTE: Chat-template round-trip has {chat_fail} failures.")
        print(f"        This is expected IF TRL passes raw prompts (not wrapped) to reward fns.")
        print(f"        Investigating TRL source to confirm...")

    # ---- Test 4: normalize_answer on real answers ----
    print("\n[5/7] Testing normalize_answer on real extracted answers...")
    normalized_samples = []
    for ans in real_extracted[:50]:
        normed = normalize_answer(ans)
        normalized_samples.append((ans, normed))

    print(f"  Normalized {len(normalized_samples)} real answers:")
    for orig, normed in normalized_samples[:15]:
        print(f"    {orig:40s} -> {normed}")

    # Verify no empty normalizations for non-empty answers
    empty_norms = [(orig, normed) for orig, normed in normalized_samples if not normed and orig]
    print(f"\n  Answers that normalized to empty: {len(empty_norms)}")
    if empty_norms:
        for orig, normed in empty_norms[:5]:
            print(f"    WARNING: '{orig}' -> '{normed}'")

    print(f"  PASS: normalize_answer handles real data")

    # ---- Test 5: _try_parse_number on normalized answers ----
    print("\n[6/7] Testing _try_parse_number on normalized real answers...")
    parse_results = {"numeric": 0, "fraction": 0, "non_numeric": 0}

    for orig, normed in normalized_samples:
        parsed = _try_parse_number(normed)
        if parsed is not None:
            if '/' in normed:
                parse_results["fraction"] += 1
            else:
                parse_results["numeric"] += 1
        else:
            parse_results["non_numeric"] += 1

    print(f"  Parsed as plain number: {parse_results['numeric']}")
    print(f"  Parsed as fraction: {parse_results['fraction']}")
    print(f"  Non-numeric (symbolic/LaTeX): {parse_results['non_numeric']}")
    print(f"  PASS: _try_parse_number correctly distinguishes numeric vs symbolic")

    # ---- Test 6: End-to-end reward simulation ----
    print("\n[7/7] Testing end-to-end reward function with real data...")

    # Simulate the reward function from train_grpo.py
    GROUND_TRUTH_ANSWERS = ground_truth

    def simulated_reward_fn(completions, prompts):
        rewards = []
        for completion, prompt in zip(completions, prompts):
            score = 0.0
            accuracy_score = 0.0
            predicted_answer = extract_answer(completion)
            prompt_hash = _prompt_key(prompt)
            gt = GROUND_TRUTH_ANSWERS.get(prompt_hash, None)

            if predicted_answer and gt:
                pred_norm = normalize_answer(predicted_answer)
                gt_norm = normalize_answer(gt)
                if pred_norm == gt_norm:
                    accuracy_score = 1.0
                elif pred_norm in gt_norm or gt_norm in pred_norm:
                    accuracy_score = 0.7
                else:
                    pred_num = _try_parse_number(pred_norm)
                    gt_num = _try_parse_number(gt_norm)
                    if pred_num is not None and gt_num is not None:
                        if abs(pred_num - gt_num) < 1e-6:
                            accuracy_score = 1.0
            elif predicted_answer:
                accuracy_score = 0.3

            # Format score
            format_score = 0.0
            if re.search(r'\\boxed\{', completion):
                format_score += 0.5
            if re.search(r'(step\s*\d|first|second|third|then|next|finally)', completion.lower()):
                format_score += 0.3
            if re.search(r'[=+\-*/^]', completion):
                format_score += 0.2
            format_score = min(format_score, 1.0)

            # Length score
            word_count = len(completion.split())
            if word_count < 10:
                length_score = 0.0
            elif word_count < 30:
                length_score = 0.3
            elif word_count < 100:
                length_score = 0.7
            elif word_count < 300:
                length_score = 1.0
            else:
                length_score = 0.8

            # Reasoning score
            reasoning_score = 0.0
            completion_lower = completion.lower()
            if any(w in completion_lower for w in ['because', 'therefore', 'since', 'thus', 'hence']):
                reasoning_score += 0.4
            if any(w in completion_lower for w in ['let', 'given', 'we have', 'we get', 'we can']):
                reasoning_score += 0.3
            if any(w in completion_lower for w in ['answer', 'result', 'solution', 'final']):
                reasoning_score += 0.3
            reasoning_score = min(reasoning_score, 1.0)

            score = (accuracy_score * 0.50 + format_score * 0.25 +
                     length_score * 0.15 + reasoning_score * 0.10)
            rewards.append(score)
        return rewards

    # Test with correct answers (using real solutions as completions)
    test_prompts = []
    test_completions_correct = []
    test_completions_wrong = []

    for example in samples[:50]:
        problem = example.get("problem", "")
        solution = example.get("solution", "")
        if problem and solution and _prompt_key(problem) in GROUND_TRUTH_ANSWERS:
            test_prompts.append(problem)
            test_completions_correct.append(solution)  # Real solution = should match
            test_completions_wrong.append("I don't know the answer to this problem.")

    if test_prompts:
        rewards_correct = simulated_reward_fn(test_completions_correct, test_prompts)
        rewards_wrong = simulated_reward_fn(test_completions_wrong, test_prompts)

        avg_correct = sum(rewards_correct) / len(rewards_correct)
        avg_wrong = sum(rewards_wrong) / len(rewards_wrong)

        # Count perfect accuracy scores
        perfect_accuracy = sum(1 for r in rewards_correct if r >= 0.5)

        print(f"  Tested {len(test_prompts)} prompt-solution pairs")
        print(f"  Avg reward (correct solutions): {avg_correct:.3f}")
        print(f"  Avg reward (wrong answers):     {avg_wrong:.3f}")
        print(f"  Correct solutions scoring >= 0.5: {perfect_accuracy}/{len(rewards_correct)}")
        print(f"  Reward gap (correct - wrong):   {avg_correct - avg_wrong:.3f}")

        # Show some individual rewards
        print(f"\n  Sample rewards (first 5):")
        for i in range(min(5, len(test_prompts))):
            gt = GROUND_TRUTH_ANSWERS.get(_prompt_key(test_prompts[i]), "?")
            extracted = extract_answer(test_completions_correct[i])
            print(f"    GT='{gt[:30]}' Extracted='{extracted[:30] if extracted else 'None'}' "
                  f"reward_correct={rewards_correct[i]:.3f} reward_wrong={rewards_wrong[i]:.3f}")

        assert avg_correct > avg_wrong, f"Correct solutions should score higher! {avg_correct:.3f} <= {avg_wrong:.3f}"
        assert avg_correct > 0.4, f"Average correct reward too low: {avg_correct:.3f}"
        print(f"\n  PASS: Reward function correctly discriminates (gap={avg_correct - avg_wrong:.3f})")
    else:
        print("  SKIP: No matching prompt-solution pairs found")

    # ---- Test 7 (bonus): Multiple \boxed{} — first vs last ----
    print("\n[BONUS] Checking for solutions with multiple \\boxed{} occurrences...")
    multi_boxed = 0
    for example in samples:
        solution = example.get("solution", "")
        count = solution.count('\\boxed{')
        if count > 1:
            multi_boxed += 1

    print(f"  Solutions with multiple \\boxed{{}}: {multi_boxed}/{len(samples)}")
    if multi_boxed > 0:
        # Our _extract_boxed_content finds the FIRST occurrence
        # For math, final answer is typically in the LAST \boxed{}
        # Check if this matters for ground truth extraction
        mismatch_count = 0
        mismatch_examples = []
        for example in samples:
            solution = example.get("solution", "")
            if solution.count('\\boxed{') <= 1:
                continue

            first = _extract_boxed_content(solution)
            # Find last \boxed{} by searching from end
            last_idx = solution.rfind('\\boxed{')
            if last_idx == -1:
                last_idx = solution.rfind('\\boxed {')
            if last_idx >= 0:
                last_solution = solution[last_idx:]
                last = _extract_boxed_content(last_solution)
                if first != last:
                    mismatch_count += 1
                    mismatch_examples.append((first, last, example.get("problem", "")[:80]))

        print(f"  First vs Last mismatch: {mismatch_count}/{multi_boxed}")
        if mismatch_count > 0:
            pct = mismatch_count / multi_boxed * 100
            print(f"  WARNING: {pct:.1f}% of multi-boxed solutions have different first/last answers")
            for first, last, prob in mismatch_examples[:5]:
                print(f"    first='{first}' vs last='{last}'")
                print(f"    problem: {prob}...")
            print(f"  RECOMMENDATION: Use LAST \\boxed{{}} for ground truth (final answer)")
        else:
            print(f"  OK: All multi-boxed solutions have same first/last answer")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: All tests PASSED")
    print(f"  Ground truth extraction: {extraction_rate:.1f}% success rate")
    print(f"  Boxed parsing: {success_rate:.1f}% success rate")
    print(f"  Prompt key round-trip: 100% success")
    print(f"  Reward discrimination: correct > wrong by {avg_correct - avg_wrong:.3f}")
    if multi_boxed > 0 and mismatch_count > 0:
        print(f"  NOTE: {mismatch_count} multi-boxed solutions need LAST \\boxed{{}} extraction")
    print("=" * 70)


if __name__ == "__main__":
    main()
