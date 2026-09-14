# Nota sobre el conjunto retenido (held-out) — v1.2

## Estado: ejecutado

El held-out se ejecutó el 2026-09-13 con 20 repeticiones para los diez modelos
acordados. No volver a ejecutarlo ni usar sus resultados para ajustar prompt,
modelos, thresholds o parámetros. Los resultados y el manifiesto reproducible
están en `results/judge_eval_v1_2/heldout_20rep_20260913/`.

El dataset utilizado fue `held_out_evaluation.jsonl`, con SHA-256:

```text
77fb912bb766d0745328a01f291ce65bbf28c8e83e7acc295e5af8ba85c57791
```

El spec seguía declarando `currently_frozen=false` al ejecutar el lote. Esta
limitación está registrada en el reporte y no se ocultó modificando el snapshot.
Ver también `MODEL_EVALUATION_README.md` y `judge_definition.md`.
