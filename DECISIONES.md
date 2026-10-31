# DECISIONES

Formato: decisión, alternativas consideradas, justificación.

## 1. Disponibilidad de la información en el tiempo

**Decisión:** cada archivo de la bandeja tiene `llegada_ts` en `manifest.csv`. Toda lectura es "as-of corte" (`inventario.lectura.leer_asof`): solo `llegada_ts <= corte`, y por (fuente, fecha_negocio) se usa la última versión **válida** disponible a ese corte. Cada fila lleva `available_at` y `version`.

- Un `corte` sin hora se interpreta como **fin de ese día** (23:59:59): información disponible al cierre.
- Rezagos simulados: ventas y transacciones D+1 06:00, petróleo D+2, feriados 30 días antes.
- Feriados son futuro-conocido legítimo solo si ya llegaron al corte.
- Alternativa rechazada: filtrar solo por fecha de negocio. Ignora rezagos y correcciones, y produce fuga de información.

### Features sin fuga de información
- Un pronóstico con corte C se emite desde el **origen** = último día cuyo archivo ya llegó (C menos 1 día 6 h, a fecha). Toda feature usa solo datos con fecha <= origen.
- Enfoque **directo multi-horizonte**: una fila por (origen, horizonte h, serie) con `h` como feature. Las medias móviles (7, 28, 91 días) terminan en el origen. Los rezagos "mismo día de la semana" usan la semana mínima k = ceil(h/7), así el rezago siempre cae en o antes del origen (lag >= horizonte).
- `onpromotion` de días futuros **no** se usa (no llega por adelantado en la bandeja); solo el promedio de promociones de los últimos 7 y 28 días.
- Futuro-conocido legítimo: calendario (día de la semana, día del mes, quincena/fin de mes) y feriados nacionales **que ya llegaron al corte** (llegan 30 días antes). Se ignoran feriados regionales y locales (simplificación).
- Probado: envenenar los datos posteriores al origen no cambia ninguna feature.

## 2. Simulación del sistema de origen

**Decisión:** carga histórica hasta 2016-12-31; el resto se reparte en archivos por fecha y fuente. Correcciones: una v1 distorsionada llega primero y una v2 con el dato real llega 3 a 10 días después. Generación determinista (semilla fija). Con `--anomalias` se inyectan archivos defectuosos y se registran en `data/anomalias_verdad.csv` (fuera de la bandeja) para probar la validación.

## 3. Validación de datos

**Decisión:** cada archivo (todas sus versiones) se valida al ingerirse (`inventario.validacion`).

- **Inválido → cuarentena y falla la tarea:** columnas faltantes, archivo vacío o ilegible, nulos en columnas clave, valores no numéricos, negativos, duplicados por llave, fecha del contenido distinta a la del archivo.
- **Esperado del negocio → se registra como advertencia, no falla:** petróleo faltante, día completo con ventas en cero, 25 de diciembre sin ventas (`dias_sin_operacion` en `fuentes.yaml`).
- **Fechas sin archivo:** se reportan como `faltantes` (pueden ser retraso), no se fallan; el modelo trabaja con lo disponible y el reporte las deja visibles.
- La bandeja nunca se modifica: lo inválido se **copia** a `data/cuarentena/` junto con un `.motivo.json`.

## 4. Manejo de correcciones y reprocesamiento

**Decisión:** `data/procesado/` conserva **todas las versiones válidas** (`<fecha>_v<k>.csv`), no solo la última. Así `leer_asof` reproduce exactamente lo que se sabía en cualquier corte pasado (un pronóstico del pasado no cambia cuando llega una corrección posterior).

- Si la última versión de una fecha es inválida, esa versión va a cuarentena (la tarea falla y alerta) y la lectura as-of usa la última versión válida anterior, nunca datos inválidos.
- Idempotencia: copia byte a byte con reemplazo atómico (archivo temporal + `os.replace`); re-ejecutar produce archivos idénticos. La ingesta incremental (`desde`) equivale a la completa (probado).

## 5. Costos de negocio

**Decisión:** costos en `config/costos.yaml` por familia (supuestos: alta rotación con quiebre más caro, perecederos con merma más cara). Pedido = cuantil `cu/(cu+co)` del pronóstico (newsvendor).

## 6. Modelos, validación y criterio de reemplazo

- **Horizonte:** 7 días (pedido semanal), configurable en `config/modelo.yaml`.
- **Baseline:** seasonal naive. Cuantiles empíricos del mismo día de la semana en las últimas 8 semanas. Piso de comparación que además entrega incertidumbre.
- **Modelo ML:** LightGBM con regresión cuantil (q10, q25, q50, q75, q90), un modelo global por cuantil, objetivo en log1p (los cuantiles se conservan al des-transformar). Cuantiles ordenados para evitar cruces y sin negativos.
- **Deep learning:** no se usa. El costo (más cómputo, más riesgo de reproducibilidad) no se justifica frente a un modelo de árboles con 1,782 series diarias.
- **Validación:** rolling-origin con 4 folds separados 14 días, **los mismos folds para todos los modelos**. Cada fold lee la información tal como estaba disponible a su corte (as-of, incluidas las correcciones que aún no habían llegado) y se evalúa contra la realidad final.
- **Métricas técnicas:** pinball loss (promedio de cuantiles), WAPE y sesgo con la mediana.
- **Métrica de negocio:** costo = cu·faltante + co·sobrante, con los costos de `config/costos.yaml`, sobre el pedido sugerido.
- **Pedido sugerido (newsvendor):** cuantil cu/(cu+co) del pronóstico, interpolado linealmente entre los cuantiles disponibles; fuera de [q10, q90] se acota. Pedido diario por tienda y familia.
- **Criterio de reemplazo:** el retador reemplaza al vigente si baja el costo de negocio >= 2% en al menos 3 de 4 folds **y** su pinball promedio no empeora. Parámetros en `config/modelo.yaml`. El vigente se registra en `config/vigente.json` (en Git, así el cambio de modelo pasa por PR) (por defecto: baseline).
- **Supuesto a validar:** con datos sintéticos simples el baseline empata o gana, y el criterio rechaza correctamente al retador. El resultado con Favorita real queda documentado en `reports/`.

## 7. Orquestación (Airflow 3)

- **Cuatro DAGs**, con la lógica en `src/inventario/` y los DAGs solo traduciendo el contexto de Airflow. `ingesta_validacion` (diario) publica el asset `bandeja_procesada` con el `corte` en el evento; `preparacion_datos` y `seguimiento_desempeno` se disparan por ese asset; `pronostico_pedido` es semanal.
- **Corte de cada corrida:** evento de asset (el más reciente si se acumularon varios), si no `logical_date`, si no `run_after`, si no el último día ingerido. Así el backfill reproduce la historia con el corte correcto.
- **Archivos inválidos:** la ingesta falla solo si llegaron archivos inválidos *nuevos* (ventana del último día), para que un archivo ya reportado no deje fallando todas las corridas siguientes. El pronóstico **sigue** con los archivos en cuarentena excluidos y deja el conteo en `metadata.json` (`calidad`), porque detener el pedido semanal por un solo día malo es más caro que pronosticar con el último dato válido, y queda auditable.
- **Idempotencia:** escrituras atómicas (tmp + rename) y firma de entradas; repetir una tarea produce los mismos bytes.

## 8. DVC y reproducibilidad

- Pipeline `bandeja -> ingesta -> features -> entrenamiento -> pronostico -> seguimiento`; cortes en `params.yaml`. Features y modelos se materializan con una **firma** (datos válidos hasta el corte + configuración); si cambia algo, se regeneran.
- Un pronóstico pasado se reproduce regenerándolo en un directorio temporal con la lectura as-of a su corte y comparando el SHA del CSV publicado (`verificar_pronostico.py`). Una corrección que llega después no lo altera.
- Salidas con `persist: true` para conservar cortes anteriores. El modelo vigente vive en Git (`config/vigente.json`).
- El repositorio es público para poder proteger `main` en el plan gratuito de GitHub.
- **Versiones fijadas.** Una prueba real mostró que regenerar un pronóstico con pandas 3.1.0 (local) en vez de 3.0.5 (contenedor) cambiaba el SHA del CSV, aunque LightGBM fuera el mismo. Por eso `requirements.txt` y `requirements-airflow.txt` fijan pandas, numpy y lightgbm con las versiones del contenedor.
