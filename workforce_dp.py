"""
Modelo de Tamaño de la Fuerza de Trabajo mediante Programación Dinámica
========================================================================
Problema clásico de optimización estocástica/determinística en Investigación
de Operaciones (Taha, Hamdy A. - Operations Research: An Introduction).

Este módulo resuelve el problema del contratista que planifica su fuerza
de trabajo durante un horizonte de 5 semanas minimizando los costos de
exceso de personal y costos de contratación mediante Programación Dinámica
hacia atrás (Backward Induction), aplicando principios de Programación
Orientada a Objetos (POO).

Autor: Senior IO & Python Engineer
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import sys
import argparse


# =====================================================================
# 1. PARÁMETROS DEL PROBLEMA Y COSTOS (POO)
# =====================================================================

@dataclass(frozen=True)
class ProblemParameters:
    """
    Representa los parámetros de configuración y costos del problema.
    """
    demands: Tuple[int, ...] = (5, 7, 8, 4, 6)
    excess_cost_per_worker: float = 300.0
    hiring_fixed_cost: float = 400.0
    hiring_variable_cost: float = 200.0
    firing_fixed_cost: float = 0.0
    firing_variable_cost: float = 0.0
    initial_workforce: int = 0

    @property
    def num_weeks(self) -> int:
        return len(self.demands)

    @property
    def max_demand(self) -> int:
        return max(self.demands)


class CostCalculator:
    """
    Calculadora desacoplada encargada de la lógica de evaluación de costos.
    Cumple con el principio de Responsabilidad Única (SRP).
    """
    def __init__(self, params: ProblemParameters):
        self.params = params

    def calculate_excess_cost(self, workforce: int, demand: int) -> float:
        """Costo por mantener trabajadores excedentes en la semana."""
        excess = max(0, workforce - demand)
        return excess * self.params.excess_cost_per_worker

    def calculate_hiring_cost(self, current_workforce: int, previous_workforce: int) -> float:
        """Costo por contratación si la fuerza de trabajo aumenta."""
        hired = max(0, current_workforce - previous_workforce)
        if hired > 0:
            return self.params.hiring_fixed_cost + (hired * self.params.hiring_variable_cost)
        return 0.0

    def calculate_firing_cost(self, current_workforce: int, previous_workforce: int) -> float:
        """Costo por despido/reducción de personal (si aplica)."""
        fired = max(0, previous_workforce - current_workforce)
        if fired > 0 and (self.params.firing_fixed_cost > 0 or self.params.firing_variable_cost > 0):
            return self.params.firing_fixed_cost + (fired * self.params.firing_variable_cost)
        return 0.0

    def evaluate_transition(
        self,
        prev_state: int,
        decision: int,
        demand: int,
        future_cost: float
    ) -> TransitionEvaluation:
        """Evalúa por completo una transición (x_{i-1} -> x_i)."""
        excess = max(0, decision - demand)
        hired = max(0, decision - prev_state)
        fired = max(0, prev_state - decision)

        c_excess = self.calculate_excess_cost(decision, demand)
        c_hire = self.calculate_hiring_cost(decision, prev_state)
        c_fire = self.calculate_firing_cost(decision, prev_state)
        stage_cost = c_excess + c_hire + c_fire
        total_cost = stage_cost + future_cost

        return TransitionEvaluation(
            prev_state=prev_state,
            decision=decision,
            demand=demand,
            excess_workers=excess,
            hired_workers=hired,
            fired_workers=fired,
            excess_cost=c_excess,
            hiring_cost=c_hire,
            firing_cost=c_fire,
            immediate_stage_cost=stage_cost,
            future_cost=future_cost,
            total_cost=total_cost
        )


# =====================================================================
# 2. MODELADO DE ESTADOS, DECISIONES Y ETAPAS
# =====================================================================

@dataclass
class TransitionEvaluation:
    """Detalle cuantitativo de evaluar una alternativa de decisión en un estado."""
    prev_state: int             # x_{i-1}
    decision: int               # x_i
    demand: int                 # b_i
    excess_workers: int         # max(0, x_i - b_i)
    hired_workers: int          # max(0, x_i - x_{i-1})
    fired_workers: int          # max(0, x_{i-1} - x_i)
    excess_cost: float          # C1
    hiring_cost: float          # C2
    firing_cost: float          # C3
    immediate_stage_cost: float # C1 + C2 + C3
    future_cost: float          # f_{i+1}(x_i)
    total_cost: float           # Inmediato + Futuro


@dataclass
class StateDecisionSummary:
    """Resultado óptimo para un estado particular en una etapa."""
    state: int
    optimal_decision: int
    min_cost: float
    evaluations: List[TransitionEvaluation] = field(default_factory=list)
    alternative_decisions: List[int] = field(default_factory=list)


@dataclass
class Stage:
    """Representa una semana (etapa) en la Programación Dinámica."""
    stage_number: int           # i in [1, 2, ..., N]
    demand: int                 # b_i
    feasible_states: List[int]  # Estados x_{i-1} considerados
    state_results: Dict[int, StateDecisionSummary] = field(default_factory=dict)


@dataclass
class ScheduleStep:
    """Paso óptimo reconstruido para la presentación del plan de acción semanal."""
    week: int
    initial_workforce: int
    optimal_workforce: int
    demand: int
    excess: int
    hired: int
    fired: int
    excess_cost: float
    hiring_cost: float
    firing_cost: float
    week_cost: float
    cumulative_cost: float


@dataclass
class OptimalSolution:
    """Solución completa del modelo."""
    parameters: ProblemParameters
    stages: List[Stage]
    schedule: List[ScheduleStep]
    total_min_cost: float


# =====================================================================
# 3. SOLUCIONADOR DE PROGRAMACIÓN DINÁMICA (ALGORITMO HACIA ATRÁS)
# =====================================================================

class WorkforceDPSolver:
    """
    Resuelve el problema de programación dinámica utilizando inducción hacia atrás.
    """
    def __init__(self, parameters: Optional[ProblemParameters] = None):
        self.params = parameters or ProblemParameters()
        self.cost_calc = CostCalculator(self.params)
        self.stages: List[Stage] = []

    def solve(self) -> OptimalSolution:
        """
        Ejecuta el algoritmo recursivo de Programación Dinámica y la reconstrucción.
        """
        n = self.params.num_weeks
        max_demand = self.params.max_demand

        # -------------------------------------------------------------
        # Determinación de espacios de estado y decisión por etapa
        # Siguiendo la teoría de Hamdy A. Taha:
        # En la etapa 1: el estado inicial es x_0.
        # En la etapa i > 1: el estado x_{i-1} puede ser cualquiera de
        # las decisiones factibles tomadas en la etapa i - 1.
        # Dado que x_i >= b_i y x_i <= max(b), x_i in [b_i, max(b)].
        # En la última etapa n: b_n <= x_n <= max(b), siendo óptimo x_n = b_n.
        # -------------------------------------------------------------
        stage_feasible_states: Dict[int, List[int]] = {}
        stage_feasible_decisions: Dict[int, List[int]] = {}

        # Etapa 1: Estado inicial
        stage_feasible_states[1] = [self.params.initial_workforce]
        stage_feasible_decisions[1] = list(range(self.params.demands[0], max_demand + 1))

        # Etapas 2 a n
        for i in range(2, n + 1):
            prev_demand = self.params.demands[i - 2]
            curr_demand = self.params.demands[i - 1]
            # Los estados posibles que llegan a la semana i son las decisiones de la semana i-1
            stage_feasible_states[i] = list(range(prev_demand, max_demand + 1))
            stage_feasible_decisions[i] = list(range(curr_demand, max_demand + 1))

        # -------------------------------------------------------------
        # Inducción hacia atrás: de etapa n hasta etapa 1
        # f_{n+1}(x_n) = 0
        # f_i(x_{i-1}) = min_{x_i >= b_i} [ Costo_Inmediato + f_{i+1}(x_i) ]
        # -------------------------------------------------------------
        f_next: Dict[int, float] = {x: 0.0 for x in range(0, max_demand + 1)}
        computed_stages: List[Stage] = []

        for i in range(n, 0, -1):
            demand = self.params.demands[i - 1]
            states = stage_feasible_states[i]
            decisions = stage_feasible_decisions[i]

            stage = Stage(stage_number=i, demand=demand, feasible_states=states)
            f_current: Dict[int, float] = {}

            for x_prev in states:
                best_cost = float('inf')
                best_decisions: List[int] = []
                evaluations: List[TransitionEvaluation] = []

                for x_curr in decisions:
                    future = f_next.get(x_curr, 0.0)
                    eval_res = self.cost_calc.evaluate_transition(
                        prev_state=x_prev,
                        decision=x_curr,
                        demand=demand,
                        future_cost=future
                    )
                    evaluations.append(eval_res)

                    if eval_res.total_cost < best_cost - 1e-9:
                        best_cost = eval_res.total_cost
                        best_decisions = [x_curr]
                    elif abs(eval_res.total_cost - best_cost) <= 1e-9:
                        best_decisions.append(x_curr)

                summary = StateDecisionSummary(
                    state=x_prev,
                    optimal_decision=best_decisions[0],
                    min_cost=best_cost,
                    evaluations=evaluations,
                    alternative_decisions=best_decisions
                )
                stage.state_results[x_prev] = summary
                f_current[x_prev] = best_cost

            f_next = f_current
            computed_stages.append(stage)

        # Invertir para ordenar las etapas de 1 a n
        self.stages = list(reversed(computed_stages))

        # -------------------------------------------------------------
        # Reconstrucción hacia adelante (Traceback) de la política óptima
        # -------------------------------------------------------------
        schedule: List[ScheduleStep] = []
        current_state = self.params.initial_workforce
        cumulative_cost = 0.0

        for stage in self.stages:
            res = stage.state_results[current_state]
            opt_decision = res.optimal_decision

            # Buscar evaluación específica
            target_eval = next(e for e in res.evaluations if e.decision == opt_decision)

            cumulative_cost += target_eval.immediate_stage_cost
            step = ScheduleStep(
                week=stage.stage_number,
                initial_workforce=current_state,
                optimal_workforce=opt_decision,
                demand=stage.demand,
                excess=target_eval.excess_workers,
                hired=target_eval.hired_workers,
                fired=target_eval.fired_workers,
                excess_cost=target_eval.excess_cost,
                hiring_cost=target_eval.hiring_cost,
                firing_cost=target_eval.firing_cost,
                week_cost=target_eval.immediate_stage_cost,
                cumulative_cost=cumulative_cost
            )
            schedule.append(step)
            current_state = opt_decision

        total_min_cost = self.stages[0].state_results[self.params.initial_workforce].min_cost

        return OptimalSolution(
            parameters=self.params,
            stages=self.stages,
            schedule=schedule,
            total_min_cost=total_min_cost
        )


# =====================================================================
# 4. GENERADOR DE REPORTES Y FORMATO DE SALIDA
# =====================================================================

class ReportGenerator:
    """
    Genera reportes detallados en texto plano estructurado y formateado.
    Permite imprimir en consola y exportar a archivos .txt con alta legibilidad.
    """
    @staticmethod
    def generate_full_report(solution: OptimalSolution) -> str:
        lines: List[str] = []
        p = solution.parameters

        def divider(char="=", length=85) -> str:
            return char * length

        # -------------------------------------------------------------
        # SECCIÓN 1: PARÁMETROS INICIALES DEL PROBLEMA
        # -------------------------------------------------------------
        lines.append(divider("="))
        lines.append("     INVESTIGACIÓN DE OPERACIONES - PROGRAMACIÓN DINÁMICA DETERMINÍSTICA")
        lines.append("            MODELO DE TAMAÑO DE LA FUERZA DE TRABAJO (WORKFORCE SIZE)")
        lines.append(divider("="))
        lines.append("")
        lines.append("1. PARÁMETROS INICIALES Y CONFIGURACIÓN DEL MODELO:")
        lines.append(divider("-"))
        lines.append(f"  * Horizonte de planificación: {p.num_weeks} semanas.")
        demands_str = ", ".join([f"Semana {idx+1} (b_{idx+1}) = {d}" for idx, d in enumerate(p.demands)])
        lines.append(f"  * Requerimientos semanales (b_t):")
        for idx, d in enumerate(p.demands):
            lines.append(f"      - Semana {idx+1}: {d} trabajadores")
        lines.append("")
        lines.append(f"  * Fuerza de trabajo inicial (x_0): {p.initial_workforce} trabajadores")
        lines.append(f"  * Costo por mano de obra excedente: ${p.excess_cost_per_worker:,.2f} / trabajador / semana")
        lines.append(f"  * Costo de contratación:")
        lines.append(f"      - Costo fijo de contratación: ${p.hiring_fixed_cost:,.2f}")
        lines.append(f"      - Costo variable por trabajador contratado: ${p.hiring_variable_cost:,.2f}")
        lines.append(f"      - Fórmula: C_contratación = ${p.hiring_fixed_cost:,.0f} + ${p.hiring_variable_cost:,.0f}*(x_i - x_{{i-1}}) si x_i > x_{{i-1}}, 0 e.o.c.")
        if p.firing_fixed_cost > 0 or p.firing_variable_cost > 0:
            lines.append(f"  * Costo de despido: Fijo=${p.firing_fixed_cost:,.2f}, Variable=${p.firing_variable_cost:,.2f}")
        else:
            lines.append("  * Costo de despido / reducción de personal: $0.00 (sin costo según formulación estándar)")
        lines.append("")
        lines.append("  * Formulación Recursiva de Bellman:")
        lines.append("      f_i(x_{i-1}) = min { C_exceso(x_i - b_i) + C_contratación(x_i - x_{i-1}) + f_{i+1}(x_i) }")
        lines.append("      sujeto a: x_i >= b_i (para todo i = 1, ..., 5)")
        lines.append("      Condición de frontera: f_6(x_5) = 0")
        lines.append("")

        # -------------------------------------------------------------
        # SECCIÓN 2: DESGLOSE ETAPA POR ETAPA (TABLAS DETALLADAS)
        # -------------------------------------------------------------
        lines.append(divider("="))
        lines.append("2. DESGLOSE DETALLADO ETAPA POR ETAPA (CÁLCULOS DE PROGRAMACIÓN DINÁMICA):")
        lines.append(divider("="))
        lines.append("Nota: El proceso se calcula por inducción hacia atrás (de la Semana 5 a la Semana 1).")
        lines.append("")

        for stage in reversed(solution.stages):
            i = stage.stage_number
            b_i = stage.demand
            lines.append(divider("-"))
            lines.append(f" >>> ETAPA {i} (SEMANA {i}) | Requerimiento mínimo b_{i} = {b_i} trabajadores")
            lines.append(divider("-"))
            lines.append("  Detalle de cálculos por cada estado de entrada (x_{i-1}) y alternativas (x_i):")
            lines.append("")

            # Encabezado de tabla
            header = (
                f"  {'x_{i-1}':^7} | {'x_i':^5} | {'Exc':^4} | {'Cont':^5} | "
                f"{'C_Exc($)':^10} | {'C_Cont($)':^10} | {'f_{i+1}($)' :^11} | "
                f"{'Total($)':^10} | {'Óptimo?':^8}"
            )
            lines.append(header)
            lines.append("  " + "-" * (len(header) - 2))

            for state in stage.feasible_states:
                summary = stage.state_results[state]
                opt_decision = summary.optimal_decision

                for eval_item in summary.evaluations:
                    is_opt = " *** SI" if eval_item.decision == opt_decision else ""
                    row = (
                        f"  {eval_item.prev_state:^7} | "
                        f"{eval_item.decision:^5} | "
                        f"{eval_item.excess_workers:^4} | "
                        f"{eval_item.hired_workers:^5} | "
                        f"{eval_item.excess_cost:^10.1f} | "
                        f"{eval_item.hiring_cost:^10.1f} | "
                        f"{eval_item.future_cost:^11.1f} | "
                        f"{eval_item.total_cost:^10.1f} | "
                        f"{is_opt:<8}"
                    )
                    lines.append(row)
                lines.append("  " + "." * (len(header) - 2))

            lines.append("")
            lines.append(f"  >> Resumen de Decisiones Óptimas para la Etapa {i}:")
            for state in stage.feasible_states:
                summary = stage.state_results[state]
                alt_txt = f" (empates: {summary.alternative_decisions})" if len(summary.alternative_decisions) > 1 else ""
                lines.append(
                    f"     - Para estado x_{i-1} = {state:2d} trabajadores: "
                    f"Decisión óptima x_{i}* = {summary.optimal_decision} | "
                    f"Costo mínimo f_{i}({state}) = ${summary.min_cost:,.2f}{alt_txt}"
                )
            lines.append("")

        # -------------------------------------------------------------
        # SECCIÓN 3: RUTA U OPERACIÓN ÓPTIMA GLOBAL
        # -------------------------------------------------------------
        lines.append(divider("="))
        lines.append("3. RUTA U OPERACIÓN ÓPTIMA GLOBAL (POLÍTICA DE PLANIFICACIÓN)")
        lines.append(divider("="))
        lines.append("Avanzando desde el estado inicial x_0 = 0 mediante las decisiones óptimas obtenidas:")
        lines.append("")

        # Encabezado del plan óptimo
        sched_header = (
            f" {'Semana':^7} | {'Inicio':^7} | {'Requerido':^9} | {'Fuerza Opt':^10} | "
            f"{'Contrat':^7} | {'Despido':^7} | {'Exceso':^6} | {'C.Contrat':^10} | "
            f"{'C.Exceso':^10} | {'C.Semana':^10} | {'Acumulado':^10}"
        )
        lines.append(sched_header)
        lines.append("-" * len(sched_header))

        for step in solution.schedule:
            row = (
                f" {step.week:^7} | "
                f"{step.initial_workforce:^7} | "
                f"{step.demand:^9} | "
                f"{step.optimal_workforce:^10} | "
                f"{step.hired:^7} | "
                f"{step.fired:^7} | "
                f"{step.excess:^6} | "
                f"${step.hiring_cost:^9.2f} | "
                f"${step.excess_cost:^9.2f} | "
                f"${step.week_cost:^9.2f} | "
                f"${step.cumulative_cost:^9.2f}"
            )
            lines.append(row)
        lines.append("-" * len(sched_header))
        lines.append("")

        # Explicación narrativa de las decisiones
        lines.append("  DESGLOSE NARRATIVO DE ACCIONES SEMANALES:")
        for step in solution.schedule:
            action_desc = []
            if step.hired > 0:
                action_desc.append(f"contratar {step.hired} trabajador(es) (costo: ${step.hiring_cost:,.2f})")
            elif step.fired > 0:
                action_desc.append(f"reducir/despedir {step.fired} trabajador(es) (sin costo)")
            else:
                action_desc.append("mantener la misma plantilla (sin contrataciones)")

            if step.excess > 0:
                action_desc.append(f"mantener {step.excess} trabajador(es) excedente(s) (costo: ${step.excess_cost:,.2f})")
            else:
                action_desc.append("fuerza laboral exactamente ajustada a la demanda")

            desc = "; ".join(action_desc)
            lines.append(
                f"  * Semana {step.week}: Contar con {step.optimal_workforce} trabajadores "
                f"(requeridos: {step.demand}). Acción: {desc}. Costo semanal: ${step.week_cost:,.2f}."
            )

        lines.append("")

        # -------------------------------------------------------------
        # SECCIÓN 4: COSTO TOTAL MÍNIMO RESULTANTE
        # -------------------------------------------------------------
        lines.append(divider("="))
        lines.append("4. RESUMEN EJECUTIVO Y COSTO TOTAL MÍNIMO:")
        lines.append(divider("="))
        lines.append(f"  * Plan Óptimo de Fuerza de Trabajo: {[s.optimal_workforce for s in solution.schedule]}")
        lines.append(f"  * Costo Total Acumulado por Contratación: ${sum(s.hiring_cost for s in solution.schedule):,.2f}")
        lines.append(f"  * Costo Total Acumulado por Exceso de Personal: ${sum(s.excess_cost for s in solution.schedule):,.2f}")
        lines.append(f"  * COSTO TOTAL MÍNIMO GLOBAL (Z*): ${solution.total_min_cost:,.2f}")
        lines.append(divider("="))
        lines.append("")
        return "\n".join(lines)


# =====================================================================
# 5. PUNTO DE ENTRADA Y CLI
# =====================================================================

def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Solución al Modelo de Tamaño de la Fuerza de Trabajo mediante Programación Dinámica."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Ruta de archivo .txt opcional para guardar el reporte estructurado."
    )
    parser.add_argument(
        "--initial-workers", "-w",
        type=int,
        default=0,
        help="Cantidad inicial de trabajadores antes de la semana 1 (por defecto: 0)."
    )
    args = parser.parse_args()

    # Instanciar parámetros del problema
    params = ProblemParameters(
        demands=(5, 7, 8, 4, 6),
        excess_cost_per_worker=300.0,
        hiring_fixed_cost=400.0,
        hiring_variable_cost=200.0,
        initial_workforce=args.initial_workers
    )

    # Resolver mediante Programación Dinámica
    solver = WorkforceDPSolver(params)
    solution = solver.solve()

    # Generar reporte formateado
    report_text = ReportGenerator.generate_full_report(solution)

    # Imprimir por consola
    print(report_text)

    # Guardar en archivo si se especificó la bandera o por defecto como respaldo
    output_file = args.output or "solucion_fuerza_laboral.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n[INFO] El reporte completo ha sido exportado exitosamente a: '{output_file}'\n")
    except Exception as e:
        print(f"\n[ERROR] No se pudo guardar el archivo: {e}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
