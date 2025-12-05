"""
scheduler.py
-------------
Planificateur local (scheduler) pour organiser l’exécution des runs dans
les orchestrateurs locaux (async & parallel).

Objectifs :
- Définir des politiques de scheduling pour choisir l’ordre d'exécution :
    * round_robin
    * random
    * weighted
    * priority-based
- Offrir un Scheduler extensible pluggable dans :
    * AsyncLocalOrchestrator (local_async.py)
    * ParallelExecutor (parallel.py)

Le scheduler NE lance pas les runs, il organise simplement leur ordre.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Iterable, Optional, Callable


# ==============================================================================
# Job abstraction
# ==============================================================================

@dataclass(order=True)
class ScheduledJob:
    """
    Représente un run de simulation à planifier.

    Attributes :
        priority   : priorité (plus petit = exécuté avant)
        index      : index du run dans la liste originale
        seed_set   : seeds du run
        metadata   : informations additionnelles (optionnelles)
    """
    priority: int
    index: int
    seed_set: Iterable[Any] = field(compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)


# ==============================================================================
# Scheduler
# ==============================================================================

class Scheduler:
    """
    Scheduler générique avec différentes politiques.

    Paramètres :
        policy : str parmi {"round_robin", "random", "weighted", "priority"}
        weights : dict optionnel {index: poids}
        priorities : dict optionnel {index: priorité}
    """

    def __init__(
        self,
        policy: str = "round_robin",
        weights: Optional[Dict[int, float]] = None,
        priorities: Optional[Dict[int, int]] = None,
    ):
        self.policy = policy
        self.weights = weights or {}
        self.priorities = priorities or {}

        valid = {"round_robin", "random", "weighted", "priority"}
        if policy not in valid:
            raise ValueError(f"Policy inconnue: {policy}. Options: {valid}")

    # ------------------------------------------------------------------
    # Construction des jobs
    # ------------------------------------------------------------------

    def _build_jobs(self, seed_sets: List[Iterable[Any]]) -> List[ScheduledJob]:
        jobs = []
        for i, seeds in enumerate(seed_sets):
            prio = self.priorities.get(i, 10)  # 10 = priorité standard
            jobs.append(ScheduledJob(priority=prio, index=i, seed_set=seeds))
        return jobs

    # ------------------------------------------------------------------
    # Politiques
    # ------------------------------------------------------------------

    def _schedule_round_robin(self, jobs: List[ScheduledJob]) -> List[ScheduledJob]:
        return sorted(jobs, key=lambda j: j.index)

    def _schedule_random(self, jobs: List[ScheduledJob]) -> List[ScheduledJob]:
        shuffled = jobs[:]
        random.shuffle(shuffled)
        return shuffled

    def _schedule_weighted(self, jobs: List[ScheduledJob]) -> List[ScheduledJob]:
        """
        Weighted scheduling : plus un job a de poids, plus tôt il a 
        une chance d’apparaître.
        """
        weighted_jobs = []
        for job in jobs:
            w = self.weights.get(job.index, 1.0)
            weighted_jobs.extend([job] * int(max(1, w)))
        random.shuffle(weighted_jobs)

        # On garde la première occurrence unique par index
        seen = set()
        ordered = []
        for job in weighted_jobs:
            if job.index not in seen:
                seen.add(job.index)
                ordered.append(job)
            if len(ordered) == len(jobs):
                break
        return ordered

    def _schedule_priority(self, jobs: List[ScheduledJob]) -> List[ScheduledJob]:
        """
        Plus petite priorité = exécuté en premier.
        """
        return sorted(jobs, key=lambda j: j.priority)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(self, seed_sets: List[Iterable[Any]]) -> List[ScheduledJob]:
        """
        Applique une politique et renvoie une liste ordonnée de ScheduledJob.
        """
        jobs = self._build_jobs(seed_sets)

        if self.policy == "round_robin":
            return self._schedule_round_robin(jobs)

        if self.policy == "random":
            return self._schedule_random(jobs)

        if self.policy == "weighted":
            return self._schedule_weighted(jobs)

        if self.policy == "priority":
            return self._schedule_priority(jobs)

        # Ne devrait jamais arriver
        raise RuntimeError("Politique inconnue dans scheduler.")

