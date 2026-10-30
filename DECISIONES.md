# DECISIONES

Formato: decisión, alternativas consideradas, justificación.

## 1. Disponibilidad de la información en el tiempo

**Decisión:** cada archivo de la bandeja tiene `llegada_ts` en `manifest.csv`. Toda lectura es "as-of corte" (`inventario.lectura.leer_asof`): solo `llegada_ts <= corte`, y por (fuente, fecha_negocio) se usa la última versión **válida** disponible a ese corte. Cada fila lleva `available_at` y `version`.

- Un `corte` sin hora se interpreta como **fin de ese día** (23:59:59): información disponible al cierre.
- Rezagos simulados: ventas y transacciones D+1 06:00, petróleo D+2, feriados 30 días antes.
- Feriados son futuro-conocido legítimo solo si ya llegaron al corte.
- Alternativa rechazada: filtrar solo por fecha de negocio. Ignora rezagos y correcciones, y produce fuga de información.

_(completar: lags de features vs. horizonte)_

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

## 6. Modelos y criterio de reemplazo

_(completar)_
