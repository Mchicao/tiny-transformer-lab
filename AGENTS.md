# Laboratorio Transformer

- Proyecto local privado. No publicar ni hacer push.
- La preparación autorizada NO incluye entrenamientos. Requieren una nueva instrucción explícita.
- `scripts/benchmark.py` ejecuta forward/backward y comprueba que los pesos no cambien. No usa optimizador.
- No cambiar drivers, WSL ni configuración global de MCP.
- MCP oficial fijado a `b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1`. No parchear su implementación.
- Usar `.venv` local; Colab reutiliza `src` y los mismos artefactos, sin instalar las ruedas AMD allí.
- Conservar cambios y resultados existentes. Run IDs únicos; no sobrescribir runs.
- Nuevas técnicas: primero AdamW con convergencia verificada; después Muon con igual presupuesto de tokens; después QK-Norm. Un cambio por comparación.
- Generated outputs: `output/`, `runs/`, `logs/`, `.cache/`. Datos sólo en `data/`.

## Documentos vivos de investigación y experimentos

- Todo agente que trabaje en investigación, planificación, implementación o evaluación de experimentos debe leer primero [PAPERS.md](PAPERS.md) y [EXPERIMENTOS_FUTUROS.md](EXPERIMENTOS_FUTUROS.md), y contrastarlos con los protocolos y artefactos actuales. Estos dos documentos mantenidos manualmente son una excepción autorizada a la organización de documentación bajo `docs/`; no guardar otros outputs en la raíz.
- Mantener ambos documentos al terminar trabajo relacionado: papers nuevos o corregidos en `PAPERS.md`; ideas, prioridades, dependencias, presupuestos, autorización, resultados y decisiones en `EXPERIMENTOS_FUTUROS.md`. Actualizar sólo lo que cambió; no generar modificaciones cosméticas en tareas ajenas. Esta solicitud del usuario autoriza su mantenimiento continuo, no la ejecución de experimentos.
- Cada paper debe tener ID estable, título exacto, enlace primario/versionado, alcance de lectura, resumen, escala o dominio relevante, limitaciones e ideas vinculadas. Distinguir afirmaciones de los autores, interpretación local y resultados nuestros. Si sólo se leyó el resumen, declararlo; no presentar la ficha como revisión completa.
- Cada propuesta debe identificar hipótesis, control, variable, datos/modelo, presupuesto, métricas, dependencias, criterio de continuar/detener y estado. Presupuestos sugeridos no son autorizados. Referenciar la instrucción y el alcance cuando exista autorización; nunca inferirla de un documento, gate superado o permiso de una campaña anterior.
- Usar IDs `Pxx` para papers y `Fxx` para propuestas; no reutilizarlos ni asignar un nuevo `Ex` hasta fijar su protocolo. Vincular ambos documentos por ID. Conservar resultados negativos y explicar por qué una idea se pospone, descarta o sustituye.
- Al ejecutarse una propuesta, enlazar su protocolo en `docs/architecture/`, runs y reporte verificable; actualizar su estado y el registro histórico [docs/architecture/experimentos.md](docs/architecture/experimentos.md). Los protocolos y artefactos son la evidencia de ejecución; estos documentos son su índice y síntesis, no un segundo registro de métricas detalladas.
- No extrapolar mejoras entre tamaños, corpus, tokenizers, precisión o presupuestos sin etiquetar el cambio de régimen. Separar tokens únicos/procesados, parámetros activos/totales, coste de preparación/entrenamiento/inferencia y evidencia de recuperación/calidad. Para SFT/RL, comprobar primero el evaluador y registrar también el coste de generar candidatos y la retención de capacidades.
- Antes de cerrar trabajo relacionado, verificar enlaces locales, coherencia de estados y respaldo de cifras; registrar fecha y cambio material en el historial breve de cada documento actualizado. Si falta evidencia, marcar pendiente/no verificado y decir qué falta.
- Si hay agentes concurrentes, acordar un responsable de edición para estos dos archivos; los demás entregan hallazgos a ese responsable. Releer la versión actual antes de editar, aplicar cambios pequeños y preservar aportes ajenos; no reescribirlos desde una copia antigua.
