# NLL, perplexity y t pareado

El t pareado estudia la incertidumbre de una comparación entre métodos. La perplexity expresa cuánta incertidumbre tiene un modelo al predecir los tokens de un texto. Son conceptos diferentes: uno es una prueba estadística; el otro, una métrica del modelo.

## t pareado: comparar diferencias dentro de parejas

Una pareja contiene dos mediciones obtenidas bajo condiciones comparables. Aquí es AdamW y Muon con la misma seed, inicialización, datos y presupuesto. La siguiente pareja usa otra seed. Emparejar reduce el efecto de que una seed resulte más favorable para ambos métodos.

Para cada pareja calculamos `d = NLL_Muon - NLL_AdamW`. Un valor positivo favorece AdamW. El test estudia la hipótesis nula de que la diferencia media de la población de ejecuciones sea cero. Su estadístico es:

`t = media(d) / (desviación_estándar_muestral(d) / sqrt(n))`

El numerador es la ventaja observada. El denominador es el error estándar: la incertidumbre estimada de esa ventaja media. Una diferencia más consistente o más parejas independientes suelen aumentar la precisión. La referencia es una distribución t con `n - 1` grados de libertad. [Definición y fórmula, NIST](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm).

Nuestros resultados finales AdamW/Muon a 6.144.000 tokens:

| Seed | AdamW | Muon | Muon menos AdamW |
| --- | ---: | ---: | ---: |
| 42 | 3,192743 | 3,302172 | +0,109429 |
| 43 | 3,186252 | 3,271451 | +0,085199 |

La diferencia media es 0,097314 nats/target; desviación muestral 0,017133; error estándar 0,012115. Por tanto, `t ≈ 8,03`, pero hay sólo un grado de libertad. Bajo los supuestos del test, el p-valor bilateral es aproximadamente **0,079**: no cruza el umbral convencional de 0,05.

El p-valor representa la probabilidad, suponiendo la hipótesis nula y el modelo estadístico, de obtener un estadístico al menos tan extremo como el observado. **No** es la probabilidad de que AdamW sea mejor, de que la hipótesis nula sea verdadera ni de que el resultado sea “casualidad”. No cruzar 0,05 tampoco demuestra igualdad.

Con muestras pequeñas se necesita que la distribución poblacional de las diferencias sea aproximadamente normal; con dos parejas no podemos comprobarlo. Las parejas deben ser independientes entre sí. Los miles de tokens y checkpoints de un mismo entrenamiento no son nuevas réplicas del entrenamiento. [Supuestos de comparación pareada, OpenStax](https://openstax.org/books/statistics/pages/10-4-matched-or-paired-samples-optional).

Los mejores checkpoints son otro criterio, seleccionado usando validation: no deben sustituir retrospectivamente la métrica principal porque produzcan un resultado estadístico más favorable. Para una nueva campaña, fijar NLL final, presupuesto y análisis antes de ejecutar nuevas seeds. La cifra de 6–10 seeds es una recomendación exploratoria, no un cálculo formal de potencia ni una garantía de significancia.

## Perplexity: expresar la pérdida como incertidumbre predictiva

Para cada token real del texto, el modelo asigna una probabilidad condicionada por los tokens anteriores. La pérdida de ese token es `-ln(probabilidad_del_token_real)`: probabilidades pequeñas producen pérdidas mayores. NLL es el promedio de esas pérdidas, ponderado por el número real de tokens evaluados.

Con logaritmos naturales:

`perplexity = exp(NLL)`

Ejemplo: si las probabilidades asignadas a tres tokens reales son 1/2, 1/4 y 1/8, su NLL media es `ln(4)` y su perplexity es 4. Es el inverso de la media geométrica de esas probabilidades, no el inverso de su media aritmética.

Como referencia idealizada, asignar igual probabilidad entre cuatro opciones en cada posición produce perplexity 4. Así puede interpretarse como un número de alternativas equiprobables equivalentes; no implica que existan literalmente cuatro candidatos posibles. Menor es mejor; el mínimo ideal es 1. La tokenización y el contexto disponible afectan el resultado. [Definición y evaluación, Hugging Face](https://huggingface.co/docs/transformers/perplexity).

En seed42, AdamW final tiene `exp(3,192743) ≈ 24,355`; Muon, `exp(3,302172) ≈ 27,172`. AdamW reduce la perplexity aproximadamente 10,4 %. Una mejora de NLL de 0,052 equivale a una reducción de perplexity cercana al 5,1 %, porque `PPL_mejor / PPL_peor = exp(-0,052)`.

Estos porcentajes no indican precisión de aciertos ni mejora equivalente de coherencia. La métrica evalúa las probabilidades sobre el texto real, sin muestrear una continuación. Comparar requiere los mismos datos, tokenizer y protocolo de contexto; nuestros valores TinyStories/BPE4096 no se comparan directamente con cifras publicadas de GPT-2 o Llama en otros corpus/tokenizers. Las generaciones con prompts fijos complementan esta medición.
