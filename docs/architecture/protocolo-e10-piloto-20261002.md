# E10: piloto de búsqueda en inferencia, pesos originales fijos

Autorización autónoma hasta09:00 Santiago de2026-10-02, dentro del roadmap E10. E11 ya obtuvo un baseline de datos estable; E7 no se amplía con el LR bajo porque su gate cerró. Este piloto no adapta pesos ni instala vLLM/otra pila.

## Contrato antes de observar resultados

- Modelo: tinctura-v1 original final96M, revisión `f475b8a2698b0f2de14fbae584014908fec4f719`, densoK1, cómputoFP32. No usar pesos de E7 adaptados.
- Tareas: primeros16 ejercicios en el orden archivado de `data/posttrain/fineweb-e7-v2/arithmetic-probe.json`. Generación libre tras la pregunta; respuesta extraída como primer entero en el texto generado. Opciones de elección múltiple no se suministran. Semillas42/43/44 por método e ítem.
- Horizonte fijo24 tokens nuevos, EOS0 revisable, padding posterior determinista implícito. Suffix restart uniforme en0..23; r tras EOS es self-transition, no se reduce el horizonte por acortar una respuesta.
- Métodos: greedy y sampling base de una respuesta como referencias baratas; best-of-N de sampling base elegido por likelihood del propio modelo, power-MH de una cadena α2 y Parallel Power Tempering de dos cadenas α1/α2 como brazos de búsqueda. No usar el verificador de aritmética para seleccionar respuestas.
- Cap de los tres brazos de búsqueda: **16.000 tokens realmente procesados en forward por ítem/seed**, contando repetidos prefijos (este modelo no tiene caché KV). Registrar también llamadas,Σlongitud², tiempo y propuestas/aceptaciones. Es un cap comparable de trabajo, no igualdad exacta de FLOPs/wall; greedy/sampling single-shot no son controles compute-matched.
- Una respuesta por método/ítem/seed para exact-match del entero, pass@1.16×3=48 observaciones por método; descriptivo local y correlacionado por ítem, no significancia ni réplica de resultados4B–9B.

## Verificaciones y límites

Reutilizar los mismos pesos entre estados de cadenas. MH debe incluir probabilidades de propuesta directa/inversa; swaps reutilizan logits/likelihoods de cada estado. Antes de correr, un self-check de vocabulario finito comprueba preservación de la distribución objetivo e intercambio. Hash de pesos antes/después y RNG global aislado, sin optimizador. Fuentes/sampler congelados desde las primeras evidencias del piloto; resultados en directorios únicos.

Límite de100.000 intentos MH para evitar loops sin progreso cuando el restart cae tras EOS. Si el cap no se consume, se reporta el uso real y la causa; no se afirma igualdad de trabajo. Una propuesta que excede el cap se descarta y se devuelve el último estado aceptado completo.16k por ítem es un techo; una petición que sobrepase600s no lanza más cómputo. No iniciar nuevos lotes a partir de08:50 para preservar la hora de corte.

Gate exploratorio: registrar diferencias frente a best-of-N y single-chain a uso comparable, además de las referencias baratas. Ampliar sólo si hay señal consistente y el sampler pasa sus invariantes; si no, cerrar como límite de este96M/horizonte/sonda. No interpretar likelihood alta o textos fluidos como corrección. El piloto no valida habilidades de razonamiento general ni eficiencia sobre el stack CUDA/vLLM del paper.
