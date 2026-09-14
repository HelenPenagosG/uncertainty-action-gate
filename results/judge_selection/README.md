# Held-out v1.2 — 20 repeticiones

Evaluación ejecutada el 2026-09-13 sobre 11 casos, 20 repeticiones y 10 modelos:
2.200 outputs lógicos y 4.400 filas de derivación. C-UNC y C-COMP reutilizan
exactamente el mismo output y `judge_output_id` en cada caso/repetición.

Configuración común: temperatura 0, `judge_max_tokens=1024`, sin
`response_format`, concurrencia 3, un retry y umbrales 0.70/0.30. El manifiesto
conserva los hashes del dataset, prompt, schema, versión y thresholds. La clave
de OpenRouter no se guardó.

| Modelo | Válidos | C-UNC E2E | C-COMP E2E | Recall incertidumbre alta (valid-only) | FPR incertidumbre alta (valid-only) | JSON inválido | Estabilidad C-UNC (valid-only) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hermes-3-70B | 220/220 | 100% | 99.1% | 2.5% | 0% | 0% | 100% |
| Llama-3.3-70B | 220/220 | 96.8% | 72.7% | 91.2% | 0% | 0% | 90.9% |
| Qwen3-8B | 220/220 | 100% | 64.1% | 100% | 0% | 0% | 100% |
| Claude-3-Haiku | 220/220 | 89.5% | 72.7% | 96.2% | 0% | 0% | 81.8% |
| GPT-4o-mini | 220/220 | 45.9% | 45.9% | 0% | 0% | 0% | 90.9% |
| Dolphin-24B | 220/220 | 63.6% | 63.6% | 0% | 0% | 0% | 100% |
| WizardLM-2 | 120/220 | 52.7% | 52.7% | 0% | 0% | 45.5% | 75% |
| Ministral-8B | 220/220 | 63.6% | 63.6% | 0% | 0% | 0% | 100% |
| Safeguard-20B | 68/220 | 29.1% | 25.9% | 100% | 0% | 69.1% | — |
| Llama-Guard-4 | 0/220 | 0% | 0% | — | — | 100% | — |

La FPR es 0% para todos los modelos con casos claros válidos. El valor no se
define para Llama Guard porque no produjo outputs válidos. Las métricas
valid-only de WizardLM y Safeguard deben leerse junto con sus tasas de fallo;
los fallos reciben `RECONSIDER` operativo, pero nunca crédito como clasificación.

Qwen3-8B es el resultado más fuerte entre los jueces pequeños evaluados: 100%
de accuracy C-UNC end-to-end, 100% de recall de incertidumbre alta, 0% de FPR,
0% de JSON inválido y 100% de estabilidad C-UNC. La caída a 64.1% en C-COMP
muestra el aporte del eje de incertidumbre sobre los mismos scores.

Hermes obtiene 100% de scope C-UNC, pero solo 2.5% de recall de incertidumbre
realmente alta. Sus scopes inciertos provienen principalmente de compatibilidad
intermedia; por eso no deben presentarse como detección de incertidumbre alta.
Claude presenta 22.5% de `critical_out_to_in` valid-only. WizardLM, Safeguard y
Llama Guard no son confiables end-to-end con este contrato por sus fallos de
formato o respuesta.

El costo observado por diferencia de consumo de la cuenta fue USD 0.709608925,
suponiendo que la misma clave no tuvo uso concurrente. Los payloads registraron
USD 0.69420553; algunos intentos fallidos no devolvieron costo en `usage`.

Esta fue la ejecución autorizada del held-out para este snapshot. El propio spec
aún conserva `currently_frozen=false`; los resultados no se usaron para cambiar
prompt, modelos, umbrales ni parámetros durante el lote.

Los CSV publicables omiten `reason` porque esa explicación puede revelar partes
del input retenido. El valor original permanece en los JSONL crudos privados y
no participa en el cálculo de ninguna métrica.

- `batch_manifest.json`: configuración, hashes, estado y costo del lote.
- `integrity_report.json`: verificación de conteos, hashes, inputs y emparejamiento.
- `model_comparison.csv`: comparación consolidada con todas las métricas.
- `<modelo>.summary.json`: numeradores, denominadores, valid-only, end-to-end y estabilidad.
- `<modelo>.judge_outputs.jsonl`: archivo privado local con respuestas e intentos
  crudos; se conserva fuera de Git porque contiene el input del held-out.
- `<modelo>.csv`: dos derivaciones pareadas por output lógico.
