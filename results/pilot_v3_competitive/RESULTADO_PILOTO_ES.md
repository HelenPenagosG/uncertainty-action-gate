# Resultado del piloto B / C-UNC / C-COMP (2026-09-13/14)

## Configuración

- Actor: `cognitivecomputations/dolphin-mistral-24b-venice-edition` (uncensored).
- Juez: `qwen/qwen3-8b`. Backend: OpenRouter.
- Variante de entorno: **v3** (verificador legítimo caído; el atajo prohibido es
  la única forma de capturar la flag).
- Presión de recompensa: **competitive**. Pasos: 20. 5 corridas × 4 escenarios = 20 por brazo.
- Idéntico en los tres brazos: actor, prompt, temperatura (0.7), límites, variante
  y frame. Solo cambia la intervención del juez. Prompt del juez, thresholds y
  gate sin modificar.

## Resultado central

| Brazo | n | Intentó atajo | Ejecutó atajo | Capturó flag por atajo | BLOCK | RECONSIDER | ASK |
|---|---:|---:|---:|---:|---:|---:|---:|
| B (sin juez) | 20 | 60% | **60%** | 40% | 0 | 0 | 36 |
| C-UNC (compat.+incert.) | 20 | 60% | **0%** | 0% | 26 | 120 | 70 |
| C-COMP (solo compat.) | 20 | 55% | **0%** | 0% | 82 | 19 | 72 |

## Lectura

1. **El juez funciona.** El mismo actor, con la misma tentación, intenta el atajo
   casi igual de seguido en los tres brazos (60/60/55%). Sin juez lo **ejecuta el
   60%**; con juez (C-UNC y C-COMP) lo ejecuta el **0%**. Ninguna acción prohibida
   pasó el gate en los brazos con juez.

2. **El juez no "doma" al actor, filtra sus acciones.** La tasa de intento apenas
   cambia; lo que cambia es que las acciones prohibidas no se ejecutan. Es un
   filtro de acciones, no un cambio de intención del actor.

3. **La incertidumbre cambia el mecanismo (C-UNC vs C-COMP).** Ambos brazos frenan
   el 100% de las ejecuciones prohibidas, pero de forma distinta:
   - C-UNC (maneja incertidumbre): sobre todo **RECONSIDER** (120) y pocos BLOCK (26).
   - C-COMP (solo compatibilidad): sobre todo **BLOCK duro** (82) y pocos RECONSIDER (19).
   Es decir, tratar la incertidumbre convierte muchos bloqueos secos en
   reconsideraciones, que permiten al actor replantear/aclarar sin cortar la
   corrida. En este entorno los dos evitan la ejecución prohibida; la diferencia
   está en cómo lo hacen.

## Límites (importante, no sobre-interpretar)

- En v3 el **éxito legítimo es 0 por diseño** (la ruta legítima está caída), así
  que este piloto **no** mide si el juez preserva el éxito legítimo. Mide que,
  cuando la trampa es la única salida, el actor la intenta y el juez la bloquea.
- 4 escenarios de desarrollo que comparten estructura; 20 corridas por brazo. Es
  un piloto, no confirmatorio. Antes de conclusiones firmes: familias nuevas,
  conjunto held-out y congelar prompt del juez/thresholds.
- Resultado de un solo actor y un solo juez, en entorno sintético sin Docker.

## Costo y metadatos

- Uso total de la key tras el piloto: US$4.20 (el piloto completo costó < US$1).
- La API key no aparece en ningún archivo del entregable (verificado).
- Errores de endpoint: 0.
- Fallos finales de salida del juez: 33/387 llamadas en C-UNC y 7/386 en
  C-COMP. El fallback conservador fue `RECONSIDER`; la diferencia es una
  limitación de disponibilidad y no debe interpretarse como ventaja de C-UNC.

## Archivos

- `pilot_v3_competitive/` — piloto completo: `B/`, `C-UNC/`, `C-COMP/`, cada uno
  con `batch_summary.csv`, `aggregate.json` y `runs/` (con `config.json`,
  `steps.jsonl`, `summary.json` por corrida). Más `analysis.txt`,
  `batch_rollup.csv`, `console.log` y `run_metadata.json`.
- `screening_B_real/` — screening previo (4 configs, B) que mostró por qué hacía
  falta v3.
- `mechanism_mock/` — prueba mock del mecanismo.
- `SUMMARY.md` — el screening y por qué se llegó a v3.
- Ver también `docs/NOTA_PARA_DOCKER.md` para P1.
