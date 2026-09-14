# Resultados corregidos: ablación pareada

> Estos son los resultados históricos de calibración. El resultado vigente del
> held-out con 20 repeticiones está en
> [heldout_20rep_20260913](../../heldout_20rep_20260913/README.md).

Generados por `scripts/recompute_paired.py` sin llamadas a modelos.
Cada CSV contiene C-UNC y C-COMP derivados del mismo output, identificado por
`judge_output_id`. Los `*.summary.json` contienen métricas valid-only,
end-to-end, estabilidad y configuración de derivación.

`model_comparison.csv` resume los diez modelos. La comparación solicitada es:
Hermes 100/87.5 (-12.5 pp), Llama-3.3 97.9/75 (-22.9 pp),
Qwen3 97.9/70.8 (-27.1 pp), WizardLM 75/75 (0 pp).
Son accuracies end-to-end; WizardLM tiene 76.6/76.6 valid-only.

Incertidumbre alta sobre 18 casos UNCERTAIN: Hermes 6, Llama-3.3 17 y Qwen3 15.
Los scopes inciertos correctos son 18, 17 y 17, respectivamente.

La nueva `uncertainty_high_false_positive_rate` es 0/30 en los casos claros para
Hermes, Llama-3.3 y Qwen3 (igual en ambos brazos porque reutilizan los scores).
Los resúmenes conservan valid-only y end-to-end. En modelos con fallos debe
priorizarse valid-only y leerse junto con `invalid_json_rate`; Llama-Guard no
tiene respuestas claras válidas y su tasa valid-only es null.

Los `*.judge_outputs.jsonl` contienen scores históricos, hash de la fuente y
registro original, **no respuestas crudas recuperadas**. Declaran
`raw_available=false`, y no inventan tokens totales ni configuración histórica
que no se hubiera registrado. El límite histórico del juez tampoco se conoce y
queda como `judge_max_tokens=null`. En evaluaciones nuevas sí se fija y registra.

Los fallos de formato no reciben crédito como acierto. Llama-Guard tiene 0
outputs válidos y 0% end-to-end; sus métricas valid-only son null.
Safeguard tiene 18/48 válidos: 83.3% valid-only C-UNC y 31.2% end-to-end.
Los campos de concordancia operativa incluyendo fallback se presentan aparte.

Las definiciones y denominadores se implementan en `rhlab/judge_metrics.py`.
