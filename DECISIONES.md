# DECISIONES

Aquí explico las decisiones importantes del proyecto, qué otras opciones había y por qué escogí esta.

## 1. Qué información puede usar el modelo en cada momento

En la vida real los datos no llegan al instante: las ventas de un día llegan hasta el día siguiente. Si el modelo usara datos que en ese momento todavía no existían, los resultados saldrían demasiado buenos y no servirían de nada. Por eso cada archivo tiene anotada su hora de llegada (llegada_ts en manifest.csv), y cada vez que leo datos le digo al sistema "hasta qué momento" quiero ver (el corte). Solo se leen archivos que llegaron antes de ese corte, y de cada fecha uso la última versión válida que ya había llegado.

Si el corte no trae hora, lo tomo como el final de ese día (23:59). Los retrasos que simulé son: ventas y transacciones llegan al día siguiente a las 6am, el petróleo a los 2 días, y los feriados se conocen 30 días antes.

La otra opción era filtrar solo por la fecha de la venta. La descarté porque ignora los retrasos y las correcciones, y deja que el modelo vea el futuro sin querer.

Para las features hago lo mismo. El pronóstico sale desde un origen, que es el último día del que ya hay datos. Por ejemplo, si el corte es el 12 de mayo, las ventas del 12 llegan el 13, así que el origen es el 11 de mayo y todo se calcula con datos hasta esa fecha. Con un solo modelo predigo los 7 días siguientes, y el número de días hacia adelante entra como una columna más. Uso promedios de las últimas 1, 4 y 13 semanas aproximadamente (7, 28 y 91 días) y el valor del mismo día de la semana de semanas anteriores, siempre sin pasar el origen.

No uso las promociones de los días futuros porque en el momento de pedir no se conocen. Sí uso el calendario y los feriados nacionales que ya hayan llegado. Los feriados regionales y locales los ignoré para simplificar. Además hay un test que cambia los datos posteriores al origen y comprueba que ninguna feature se mueve.

## 2. Cómo simulé el sistema de origen

Con el CSV de Kaggle no hay forma de probar cómo llegan los datos en el tiempo, porque viene completo. Entonces tomé todo hasta el 31 de diciembre de 2016 como carga histórica, y el resto lo partí en archivos por día y por fuente, cada uno con su hora de llegada.

Para hacerlo más realista, algunos días llegan primero con un valor distorsionado (versión 1) y entre 3 y 10 días después llega la versión corregida (versión 2). Todo es determinista (semilla fija), así que sale igual cada vez. También hay una opción para meter archivos defectuosos a propósito (--anomalias), y los registro en un archivo aparte para poder probar que la validación los detecta.

## 3. Qué hago con datos malos

Cada archivo se revisa cuando entra. Si está roto (faltan columnas, está vacío, tiene valores no numéricos, negativos, duplicados, o la fecha de adentro no coincide con la del archivo), lo copio a una carpeta de cuarentena con el motivo, y la tarea falla para que alguien se entere.

Hay cosas que parecen raras pero son normales del negocio, como que el 25 de diciembre no haya ventas, un día completo con ceros o que falte el dato del petróleo en fin de semana. Esas solo quedan como advertencia y no detienen nada.

Si no llega el archivo de algún día, lo reporto como faltante pero no lo trato como error, porque puede ser solo un retraso. El modelo trabaja con lo que haya. La bandeja original nunca se modifica: lo inválido se copia, no se mueve.

## 4. Qué hago cuando llega una corrección

Guardo todas las versiones válidas de cada día, no solo la última. Así, si más adelante quiero regenerar un pronóstico del pasado, veo exactamente lo que se sabía ese día y no lo que se corrigió después. Si la última versión de un día viene mala, va a cuarentena y se usa la última versión buena anterior, nunca datos inválidos.

Las copias se hacen de forma que repetir el proceso da exactamente los mismos archivos (se escribe a un archivo temporal y luego se renombra). Correr la ingesta completa o por partes da el mismo resultado, y eso está probado.

## 5. Costos del negocio

Equivocarse no cuesta lo mismo para los dos lados. Si pido de menos pierdo ventas (costo cu), y si pido de más se queda inventario o se echa a perder (costo co). Por eso definí costos por familia de producto en config/costos.yaml. Supuse que en productos de mucha rotación faltar cuesta más, y que en perecederos sobrar cuesta más por la merma. Son supuestos míos; en una empresa real vendrían de finanzas.

El pedido sugerido es el cuantil cu/(cu+co) del pronóstico. Por ejemplo, si faltar cuesta 4 y sobrar cuesta 1, pido el cuantil 0.8, o sea, más inventario. Si cambio un costo, cambia el pedido sin tocar el código.

## 6. Modelos, validación y cuándo reemplazar

El horizonte es de 7 días porque el pedido es semanal.

El modelo base (baseline) toma lo que se vendió el mismo día de la semana en las últimas 8 semanas y saca cuantiles de esos valores. Me sirve de piso de comparación, y además da una idea de la incertidumbre.

El modelo de ML es LightGBM con regresión cuantil (q10, q25, q50, q75 y q90), un modelo por cada cuantil entrenado con todas las series juntas. Entreno sobre log1p de las ventas y luego acomodo los cuantiles para que no se crucen ni salgan negativos. No usé deep learning: necesita más cómputo, es más difícil de reproducir y para estas series un modelo de árboles funciona bien.

Para validar uso 4 cortes separados 14 días, los mismos para los dos modelos. En cada corte el modelo solo ve lo que había llegado hasta ese día (incluso con correcciones que todavía no llegaban) y se compara contra la venta real final. Mido pinball loss, WAPE y sesgo, y como métrica de negocio el costo: cu por las unidades que faltaron más co por las que sobraron.

El modelo nuevo reemplaza al vigente si baja el costo al menos 2% en 3 de los 4 cortes y su pinball promedio no empeora. El modelo vigente se guarda en config/vigente.json, que está en Git, así que cambiar de modelo siempre pasa por un Pull Request.

Resultado con los datos reales de Favorita: LightGBM bajó el costo entre 5.7% y 20.2% en los 4 cortes y el pinball promedio bajó 7.9%, así que quedó como modelo vigente. En el corte 3 el pinball de LightGBM fue un poco peor que el del baseline (19.59 contra 19.32), aunque el costo sí mejoró. El detalle está en reports/backtest_2017-06-30.csv.

## 7. Cómo se orquesta con Airflow

Armé 4 DAGs, y la lógica vive en el código del proyecto, no en los DAGs; ellos solo llaman a las funciones. La ingesta corre diario y, cuando termina, avisa con un asset (bandeja_procesada) que lleva la fecha de corte. Ese aviso dispara solo la preparación de datos y el seguimiento del desempeño. El pronóstico y pedido corre semanal.

La fecha de corte de cada corrida sale, en este orden, del aviso del asset, de la fecha lógica de la corrida (así funciona bien el backfill) o del último día ya ingerido.

La ingesta falla solo si llegaron archivos inválidos nuevos en el último día. Si no, un archivo malo viejo haría fallar todas las corridas siguientes. El pronóstico sí sigue aunque haya archivos en cuarentena, dejando fuera esos archivos y anotando cuántos fueron. Prefiero eso a dejar sin pedido la semana por un solo día malo. Todas las tareas se pueden repetir sin cambiar el resultado.

## 8. DVC y reproducibilidad

Los datos y modelos no van a Git, van a DVC. El pipeline tiene las etapas bandeja, ingesta, features, entrenamiento, pronóstico y seguimiento, y los cortes se cambian en params.yaml. Las salidas conservan los cortes anteriores, así que un corte nuevo no borra los viejos.

Las features y los modelos guardan una firma de lo que los generó (datos válidos hasta el corte y la configuración). Si algo cambia, se regeneran; si no, se reutilizan.

Para comprobar la reproducibilidad hay un script (verificar_pronostico.py) que regenera un pronóstico pasado con los datos de ese momento y compara el archivo con el publicado. Con el de 2017-07-02 salieron idénticos. Al principio me dio distinto: el entorno local tenía pandas 3.1.0 y el contenedor de Airflow 3.0.5, y eso cambiaba los decimales. Por eso fijé las versiones de pandas, numpy y lightgbm en los requirements, y con eso salió igual.

El repositorio es público porque en el plan gratuito de GitHub solo así se puede proteger la rama main.

## 9. Limitaciones que conozco

- Al despausar los DAGs, Airflow lanzó una corrida con la fecha real (2026), muy posterior a los datos (2017), y falló. Debería acotar los DAGs al periodo simulado con end_date o mantenerlos pausados fuera del backfill.
- Los costos por familia son supuestos míos, no datos reales.
- Ignoré los feriados regionales y locales.
- El remoto de DVC es una carpeta local; en un proyecto real sería un almacenamiento en la nube.
- No investigué por qué el corte 2 mejora tanto (20.2%); podría ser un efecto de calendario, pero es solo una hipótesis.
