from dataclasses import dataclass
import math

import torch


class BudgetExhausted(RuntimeError):
    pass


def mh_log_accept(old_logp, new_logp, alpha, reverse_logq, forward_logq):
    return min(0.0, alpha * (new_logp - old_logp) + reverse_logq - forward_logq)


def swap_log_accept(low_logp, high_logp, low_alpha, high_alpha):
    return min(0.0, (high_alpha - low_alpha) * (low_logp - high_logp))


@dataclass
class Record:
    tokens: list[int]
    logprobs: list[torch.Tensor]

    def logp(self):
        return sum(float(lp[token]) for token, lp in zip(self.tokens, self.logprobs, strict=True))

    def proposal_logq(self, alpha, start):
        return sum(float(alpha * lp[token] - torch.logsumexp(alpha * lp, 0))
                   for token, lp in zip(self.tokens[start:], self.logprobs[start:], strict=True))


class Sampler:
    def __init__(self, callback, prompt, seed, horizon=24, eos=0, token_budget=16000):
        self.callback = callback
        self.prompt = list(prompt)
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.horizon, self.eos, self.token_budget = horizon, eos, token_budget
        self.processed_tokens, self.context_squared, self.forward_calls = 0, 0, 0
        self.refinement_attempts, self.proposals, self.accepted, self.self_transitions = 0, 0, 0, 0
        self.swap_attempts, self.swaps_accepted = 0, 0

    def uniform(self):
        value = float(torch.rand((), generator=self.generator))
        return max(value, torch.finfo(torch.float64).tiny)

    def logits(self, prefix):
        ids = self.prompt + list(prefix)
        length = len(ids)
        if self.processed_tokens + length > self.token_budget:
            raise BudgetExhausted('Inference cap reached before next forward')
        logits = self.callback(ids).detach().cpu().double()
        if logits.ndim != 1 or not torch.isfinite(logits).all().item():
            raise RuntimeError('Nonfinite/incompatible inference logits')
        self.processed_tokens += length
        self.context_squared += length * length
        self.forward_calls += 1
        return logits.log_softmax(0)

    def generate(self, alpha=1.0, old=None, start=0, greedy=False):
        tokens = [] if old is None else list(old.tokens[:start])
        cached = [] if old is None else list(old.logprobs[:start])
        if self.eos in tokens:
            return Record(tokens, cached)
        while len(tokens) < self.horizon:
            logp = self.logits(tokens)
            token = int(logp.argmax()) if greedy else int(torch.multinomial((alpha * logp).softmax(0), 1, generator=self.generator))
            tokens.append(token)
            cached.append(logp)
            if token == self.eos:
                break
        return Record(tokens, cached)

    def refine(self, record, alpha):
        self.refinement_attempts += 1
        start = int(torch.randint(self.horizon, (), generator=self.generator))
        if start >= len(record.tokens):
            self.self_transitions += 1
            return record
        proposal = self.generate(alpha, record, start)
        self.proposals += 1
        log_accept = mh_log_accept(record.logp(), proposal.logp(), alpha,
                                  record.proposal_logq(alpha, start), proposal.proposal_logq(alpha, start))
        if math.log(self.uniform()) < log_accept:
            self.accepted += 1
            return proposal
        return record

    def swap(self, low, high, low_alpha=1.0, high_alpha=2.0):
        self.swap_attempts += 1
        if math.log(self.uniform()) < swap_log_accept(low.logp(), high.logp(), low_alpha, high_alpha):
            self.swaps_accepted += 1
            return high, low
        return low, high

    def run(self, method):
        if method == 'greedy':
            return self.generate(greedy=True)
        if method == 'standard':
            return self.generate()
        if method == 'best_of_n':
            best = self.generate()
            while self.processed_tokens < self.token_budget:
                try:
                    proposal = self.generate()
                except BudgetExhausted:
                    break
                self.proposals += 1
                if proposal.logp() > best.logp():
                    best = proposal
            return best
        if method == 'single_power':
            record = self.generate(2.0)
            while self.processed_tokens < self.token_budget and self.refinement_attempts < 100000:
                try:
                    record = self.refine(record, 2.0)
                except BudgetExhausted:
                    break
            return record
        if method == 'ppt':
            low, high = self.generate(1.0), self.generate(2.0)
            while self.processed_tokens < self.token_budget and self.refinement_attempts < 100000:
                try:
                    low = self.refine(low, 1.0)
                    high = self.refine(high, 2.0)
                except BudgetExhausted:
                    break
                low, high = self.swap(low, high)
            return high
        raise ValueError('Unknown inference method')

    def accounting(self):
        return dict(processed_forward_tokens=self.processed_tokens, forward_token_cap=self.token_budget,
                    context_length_squared_sum=self.context_squared, forward_calls=self.forward_calls,
                    refinement_attempts=self.refinement_attempts, proposals=self.proposals,
                    accepted=self.accepted, self_transitions=self.self_transitions,
                    swap_attempts=self.swap_attempts, swaps_accepted=self.swaps_accepted)
