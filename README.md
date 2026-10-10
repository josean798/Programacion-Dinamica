# Modelo de Tamaño de la Fuerza de Trabajo - Programación Dinámica

Este proyecto resuelve el problema clásico de optimización de la fuerza de trabajo (Workforce Size Model) mediante **Programación Dinámica Determinística hacia atrás (Backward Induction)**, aplicando principios de diseño de **Programación Orientada a Objetos (POO)** en Python.

---

## 1. Planteamiento del Problema

Un contratista debe planificar la cantidad de trabajadores para las siguientes 5 semanas ($b_t$):
- **Semana 1 ($b_1$):** 5 trabajadores
- **Semana 2 ($b_2$):** 7 trabajadores
- **Semana 3 ($b_3$):** 8 trabajadores
- **Semana 4 ($b_4$):** 4 trabajadores
- **Semana 5 ($b_5$):** 6 trabajadores

### Costos Asociados:
1. **Mano de obra excedente ($C_1$):** $\$300$ por trabajador por semana cuando la fuerza laboral $x_t$ supera la demanda $b_t$:
   $$C_1(x_t - b_t) = 300 \cdot (x_t - b_t)$$
2. **Costo de contratación ($C_2$):** Al incrementar el personal respecto a la semana previa ($x_t > x_{t-1}$), se incurre en un costo fijo de $\$400$ más $\$200$ por trabajador:
   $$C_2(x_t - x_{t-1}) = \begin{cases} 400 + 200 \cdot (x_t - x_{t-1}) & \text{si } x_t > x_{t-1} \\ 0 & \text{si } x_t \le x_{t-1} \end{cases}$$
3. **Costo de reducción/despido ($C_3$):** $\$0$ (según la formulación estándar de Hamdy A. Taha).

---

## 2. Formulación Matemática de Programación Dinámica

- **Etapas ($i$):** Semanas $i \in \{1, 2, 3, 4, 5\}$.
- **Estado ($x_{i-1}$):** Número de trabajadores disponibles al inicio de la semana $i$ (heredados de la semana $i-1$). Para $i=1$, el estado inicial es $x_0 = 0$.
- **Variable de Decisión ($x_i$):** Número de trabajadores a emplear durante la semana $i$, sujeto a $x_i \ge b_i$. Como el requerimiento máximo es 8, $b_i \le x_i \le 8$.
- **Ecuación Recursiva de Bellman:**
  $$f_i(x_{i-1}) = \min_{x_i \ge b_i} \left\{ C_1(x_i - b_i) + C_2(x_i - x_{i-1}) + f_{i+1}(x_i) \right\}$$
  con condición de frontera $f_6(x_5) = 0$.

---

## 3. Estructura del Código (POO)

El archivo [`workforce_dp.py`](file:///d:/Escritorio/MIS%20COSAS/Programacion/Tarea%20Argenis/workforce_dp.py) implementa una arquitectura modular y limpia:
- `ProblemParameters`: Dataclass inmutable con las constantes del problema y demandas.
- `CostCalculator`: Clase encargada del cómputo desacoplado de funciones de costo (SRP).
- `TransitionEvaluation`: Representa la evaluación exhaustiva de una transición $(x_{i-1} \to x_i)$.
- `StateDecisionSummary`: Almacena la decisión óptima $x_i^*$ y el costo mínimo $f_i(x_{i-1})$ para un estado.
- `Stage`: Representa una etapa (semana) con sus estados y alternativas evaluadas.
- `WorkforceDPSolver`: Ejecuta el algoritmo de Bellman hacia atrás y reconstruye la trayectoria óptima.
- `ReportGenerator`: Formatea las tablas completas, análisis narrativo y resumen para consola y archivo `.txt`.

---

## 4. Ejecución

Para ejecutar el programa y visualizar los resultados en consola y generar el archivo [`solucion_fuerza_laboral.txt`](file:///d:/Escritorio/MIS%20COSAS/Programacion/Tarea%20Argenis/solucion_fuerza_laboral.txt):

```bash
python workforce_dp.py
```

Opciones disponibles por CLI:
```bash
python workforce_dp.py --output mi_reporte.txt --initial-workers 0
```

---

## 5. Resumen de la Solución Óptima

| Semana ($i$) | Requerido ($b_i$) | Fuerza Óptima ($x_i^*$) | Contratados | Exceso | Costo Contratación | Costo Exceso | Costo Semanal |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 5 | **5** | 5 | 0 | $\$1,400$ | $\$0$ | $\$1,400$ |
| **2** | 7 | **8** | 3 | 1 | $\$1,000$ | $\$300$ | $\$1,300$ |
| **3** | 8 | **8** | 0 | 0 | $\$0$ | $\$0$ | $\$0$ |
| **4** | 4 | **6** | 0 | 2 | $\$0$ | $\$600$ | $\$600$ |
| **5** | 6 | **6** | 0 | 0 | $\$0$ | $\$0$ | $\$0$ |

- **Plan Óptimo de Fuerza de Trabajo:** $[5, 8, 8, 6, 6]$
- **Costo Total Mínimo ($Z^*$):** **$\$3,300.00**
