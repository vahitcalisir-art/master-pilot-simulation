"""
Stackelberg Game-Based Master-Pilot Joint Decision Simulation

A discrete-event simulation framework for analysing hierarchical 
decision-making interactions during port approach and berthing operations.

The model represents the master-pilot relationship as a leader-follower
game where manoeuvring outcomes depend on the balance between operational
demand and available recovery capacity.
"""

import random
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

import simpy


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class AuthorityMode(str, Enum):
    BASELINE = "baseline"
    ADAPTIVE = "adaptive"
    PROACTIVE = "proactive"


class TugRegime(str, Enum):
    NOMINAL = "nominal"
    DELAYED = "delayed"
    DEGRADED = "degraded"


PHASES: List[Tuple[str, int]] = [
    ("channel_entry", 10),
    ("speed_reduction", 8),
    ("alignment", 8),
    ("final_approach", 10),
]


@dataclass(frozen=True)
class Scenario:
    name: str
    risk: RiskLevel
    authority: AuthorityMode
    tug: TugRegime
    seed: int


@dataclass
class VesselState:
    heading_rad: float = 0.0
    speed: float = 4.0
    cte: float = 0.0
    ukc: float = 2.0
    phase: str = "channel_entry"


@dataclass
class EnvState:
    wind_ms: float
    current_ms: float
    visibility_nm: float
    bank_effect: float


@dataclass
class OperationalState:
    tug_arrived: bool = False
    tug_effectiveness: float = 1.0
    tug_arrival_delay: float = 0.0


@dataclass
class Recorder:
    events: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, t: float, event: str, payload: Dict[str, Any]) -> None:
        self.events.append({"t": float(t), "event": event, **payload})

    def summarize(self) -> Dict[str, Any]:
        loc = sum(1 for e in self.events if e["event"] == "loss_of_control")
        overrides = sum(
            1 for e in self.events
            if e["event"] == "master_decision" and e.get("decision") == "override"
        )
        end_t = max((e["t"] for e in self.events), default=0.0)
        return {
            "loss_of_control": 1 if loc > 0 else 0,
            "master_overrides": overrides,
            "end_time": end_t,
        }


def risk_pilot(s: VesselState, env: EnvState, ops: OperationalState) -> float:
    vis_pen = 0.10 if env.visibility_nm < 1.0 else 0.0
    wind_pen = min(0.10, env.wind_ms / 150.0)
    return min(
        1.0,
        0.25 * abs(s.cte)
        + 0.20 * max(0.0, 2.0 - s.ukc)
        + 0.12 * env.current_ms
        + 0.08 * env.bank_effect
        + vis_pen
        + wind_pen,
    )


def risk_master(s: VesselState, env: EnvState, ops: OperationalState) -> float:
    tug_pen = 0.0 if ops.tug_arrived else 0.15
    eff_pen = (1.0 - ops.tug_effectiveness) * 0.25
    vis_pen = 0.12 if env.visibility_nm < 1.0 else 0.0
    return min(
        1.0,
        0.22 * abs(s.cte)
        + 0.45 * max(0.0, 2.0 - s.ukc)
        + 0.15 * env.current_ms
        + 0.10 * env.bank_effect
        + tug_pen
        + eff_pen
        + vis_pen,
    )


def manoeuvring_demand(s: VesselState, env: EnvState, ops: OperationalState) -> float:
    return (
        1.0
        + 0.35 * env.current_ms
        + 0.30 * env.bank_effect
        + 0.10 * max(0.0, s.speed - 3.0)
    )


def recovery_capacity(
    s: VesselState,
    env: EnvState,
    ops: OperationalState,
    authority: AuthorityMode,
) -> float:
    base = 1.0 + 0.35 * min(2.0, s.ukc) + 0.10 * max(0.0, 4.0 - s.speed)
    tug = (0.60 * ops.tug_effectiveness) if ops.tug_arrived else 0.0
    auth = {
        AuthorityMode.BASELINE: 0.00,
        AuthorityMode.ADAPTIVE: 0.08,
        AuthorityMode.PROACTIVE: 0.12,
    }[authority]
    vis = -0.10 if env.visibility_nm < 1.0 else 0.0
    return max(0.2, base + tug + auth + vis)


def demand_capacity_ratio(
    s: VesselState,
    env: EnvState,
    ops: OperationalState,
    authority: AuthorityMode,
) -> float:
    return manoeuvring_demand(s, env, ops) / recovery_capacity(s, env, ops, authority)


@dataclass
class PilotAgent:
    def choose_action(self, s: VesselState, env: EnvState, ops: OperationalState) -> Dict[str, float]:
        d_heading = -0.10 * s.cte
        d_heading = max(-0.30, min(0.30, d_heading))
        d_speed = -0.08 if abs(s.cte) > 1.5 else 0.00
        if s.ukc < 1.2:
            d_speed -= 0.05
        return {"d_heading": d_heading, "d_speed": d_speed}


@dataclass
class MasterAgent:
    authority: AuthorityMode

    def threshold(self, s: VesselState, env: EnvState, ops: OperationalState) -> float:
        base = {
            AuthorityMode.BASELINE: 0.78,
            AuthorityMode.ADAPTIVE: 0.72,
            AuthorityMode.PROACTIVE: 0.62,
        }[self.authority]

        if self.authority == AuthorityMode.ADAPTIVE:
            if s.ukc < 1.2:
                base -= 0.05
            if not ops.tug_arrived:
                base -= 0.04

        if env.visibility_nm < 1.0:
            base -= 0.03

        return max(0.45, min(0.85, base))

    def respond(self, s: VesselState, env: EnvState, ops: OperationalState, pilot_action: Dict[str, float]) -> str:
        rm = risk_master(s, env, ops)
        th = self.threshold(s, env, ops)
        if rm <= th:
            return "accept"
        if rm <= min(0.95, th + 0.10):
            return "challenge"
        return "override"

    def apply(
        self,
        s: VesselState,
        env: EnvState,
        ops: OperationalState,
        pilot_action: Dict[str, float],
        decision: str,
    ) -> Dict[str, float]:
        if decision == "accept":
            return pilot_action
        if decision == "challenge":
            return {
                "d_heading": max(-0.22, min(0.22, pilot_action["d_heading"] * 1.15)),
                "d_speed": pilot_action["d_speed"] - 0.06,
            }
        return {
            "d_heading": max(-0.35, min(0.35, -0.18 * s.cte)),
            "d_speed": -0.15 if s.speed > 2.5 else -0.05,
        }


def tug_process(env_sim: simpy.Environment, scenario: Scenario, ops: OperationalState, rec: Recorder):
    if scenario.tug == TugRegime.NOMINAL:
        delay = random.uniform(0.0, 1.5)
        eff = random.uniform(0.85, 1.00)
    elif scenario.tug == TugRegime.DELAYED:
        delay = random.uniform(2.0, 4.0)
        eff = random.uniform(0.75, 0.95)
    else:
        delay = random.uniform(3.0, 6.0)
        eff = random.uniform(0.45, 0.75)

    ops.tug_arrival_delay = delay
    yield env_sim.timeout(delay)
    ops.tug_arrived = True
    ops.tug_effectiveness = eff
    rec.log(env_sim.now, "tug_arrived", {"eff": eff, "delay": delay})


def make_environment(risk: RiskLevel) -> EnvState:
    if risk == RiskLevel.LOW:
        return EnvState(
            wind_ms=random.uniform(2, 7),
            current_ms=random.uniform(0.2, 0.8),
            visibility_nm=random.uniform(2.0, 8.0),
            bank_effect=random.uniform(0.1, 0.5),
        )
    if risk == RiskLevel.MODERATE:
        return EnvState(
            wind_ms=random.uniform(6, 12),
            current_ms=random.uniform(0.9, 1.5),
            visibility_nm=random.uniform(0.8, 3.0),
            bank_effect=random.uniform(0.6, 1.0),
        )
    return EnvState(
        wind_ms=random.uniform(10, 18),
        current_ms=random.uniform(1.6, 2.4),
        visibility_nm=random.uniform(0.3, 1.5),
        bank_effect=random.uniform(1.0, 1.6),
    )


def run_simulation(scenario: Scenario) -> Dict[str, Any]:
    random.seed(scenario.seed)

    env_sim = simpy.Environment()
    rec = Recorder()

    s = VesselState()
    env_state = make_environment(scenario.risk)
    ops = OperationalState()

    pilot = PilotAgent()
    master = MasterAgent(authority=scenario.authority)

    env_sim.process(tug_process(env_sim, scenario, ops, rec))

    def step_phase(phase: str, duration_min: int):
        s.phase = phase
        for _ in range(duration_min):
            a_p = pilot.choose_action(s, env_state, ops)
            decision = master.respond(s, env_state, ops, a_p)
            a_final = master.apply(s, env_state, ops, a_p, decision)

            rp = risk_pilot(s, env_state, ops)
            rm = risk_master(s, env_state, ops)
            rho = demand_capacity_ratio(s, env_state, ops, scenario.authority)

            rec.log(env_sim.now, "risk_snapshot", {"risk_pilot": rp, "risk_master": rm, "rho": rho})
            rec.log(env_sim.now, "master_decision", {"decision": decision, "theta": master.threshold(s, env_state, ops)})

            s.heading_rad += a_final["d_heading"]
            s.speed = max(0.5, s.speed + a_final["d_speed"])

            lateral_drift = env_state.current_ms * 0.15 + env_state.bank_effect * 0.05
            steering_recovery = math.sin(s.heading_rad) * 0.25

            tug_recovery = 0.0
            if ops.tug_arrived:
                tug_recovery = ops.tug_effectiveness * 0.10 * min(3.0, abs(s.cte))
                tug_recovery *= (1.0 if s.cte > 0 else -1.0)

            s.cte += lateral_drift - steering_recovery - tug_recovery
            ukc_decay = 0.009 * s.speed + 0.004 * env_state.bank_effect
            if ops.tug_arrived:
                ukc_decay *= (1.0 - 0.20 * ops.tug_effectiveness)
            s.ukc = max(0.2, s.ukc - ukc_decay)

            if abs(s.cte) > 2.0 or s.ukc < 0.7:
                rec.log(env_sim.now, "near_miss", {"cte": s.cte, "ukc": s.ukc, "phase": s.phase})
            if abs(s.cte) > 3.0 or s.ukc < 0.3:
                rec.log(env_sim.now, "loss_of_control", {"cte": s.cte, "ukc": s.ukc, "phase": s.phase})
                return

            yield env_sim.timeout(1)

    for ph, dur in PHASES:
        env_sim.process(step_phase(ph, dur))
        env_sim.run(until=env_sim.now + dur)
        if any(e["event"] == "loss_of_control" for e in rec.events):
            break

    rec.log(env_sim.now, "end", {"scenario": scenario.name})
    summary = rec.summarize()
    summary.update({
        "scenario": scenario.name,
        "risk": scenario.risk.value,
        "authority": scenario.authority.value,
        "tug": scenario.tug.value,
    })
    return summary


def run_experiments(base_seed: int = 42, replications: int = 50) -> List[Dict[str, Any]]:
    results = []
    idx = 0
    for r in RiskLevel:
        for a in AuthorityMode:
            for t in TugRegime:
                for k in range(replications):
                    sc = Scenario(
                        name=f"{r.value}-{a.value}-{t.value}-rep{k}",
                        risk=r, authority=a, tug=t,
                        seed=base_seed + idx * 1000 + k
                    )
                    results.append(run_simulation(sc))
                idx += 1
    return results


if __name__ == "__main__":
    import json
    from collections import defaultdict

    print("Running simulation experiments...")
    results = run_experiments(base_seed=42, replications=50)
    
    with open("sim_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    agg = defaultdict(lambda: {"n": 0, "loc": 0, "ov": 0})
    for r in results:
        key = (r["risk"], r["authority"], r["tug"])
        agg[key]["n"] += 1
        agg[key]["loc"] += r["loss_of_control"]
        agg[key]["ov"] += r["master_overrides"]

    print(f"\nTotal: {len(results)} simulations")
    print("-" * 60)
    for key in sorted(agg.keys()):
        d = agg[key]
        print(f"{key[0]:<10} {key[1]:<12} {key[2]:<10} LoC: {100*d['loc']/d['n']:5.1f}%")
    print("-" * 60)
    print("Results saved to sim_results.json")
