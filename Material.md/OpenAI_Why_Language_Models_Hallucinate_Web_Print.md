---
source: OpenAI_Why_Language_Models_Hallucinate_Web_Print.pdf
converter: PyMuPDF4LLM
converted: 2026-11-06T20:00:55.577088
tokens_approx: 2446
---

5 de septiembre de 2025 Investigación Publicación

# Por qué los modelos de lenguaje alucinan

Leer el artículo

Escucha el artículo 11:20 Compartir

En OpenAI, hacemos todo lo posible para que los sistemas de IA sean más útiles y confiables. Aunque los modelos de lenguaje son cada vez más capaces, aún existe un gran desafío difícil de resolver por completo: las alucinaciones. Nos referimos a las instancias en las que un modelo genera con toda confianza una respuesta que no es verdadera. Nuestro nuevo artículo de investigación sostiene que los modelos de lenguaje alucinan porque los procedimientos estándar de entrenamiento y evaluación valoran más las conjeturas que el reconocimiento de la incertidumbre.

ChatGPT también alucina. GPT‑5 genera muchas menos alucinaciones, especialmente al razonar , pero aún así sigue generándolas. Las alucinaciones siguen siendo un gran desafío para

todos los grandes modelos de lenguaje, pero hacemos todo lo posible por reducirlas aún más.

## ¿Qué son las alucinaciones?

Las alucinaciones son declaraciones verosímiles pero falsas, generadas por los modelos de lenguaje. Pueden aparecer de maneras sorprendentes, incluso en respuestas aparentemente sencillas. Por ejemplo, cuando le preguntamos a un chatbot muy usado por el título de la tesis doctoral de Adam Tauman Kalai (uno de los autores de este artículo), nos dio con total seguridad tres respuestas diferentes, ninguna de ellas correcta. Cuando le preguntamos por su fecha de nacimiento, nos dio tres fechas diferentes, todas incorrectas.

## Entrenamiento enfocado en aprobar la prueba

Las alucinaciones persisten en parte porque los métodos de evaluación actuales establecen los incentivos erróneos. Si bien las evaluaciones en sí mismas no son la causa directa de las alucinaciones, la mayoría de las ellas miden el rendimiento del modelo de una manera que fomenta las conjeturas en lugar de la honestidad acerca de la incertidumbre.

Piensa en ello como si fuera una prueba de opciones múltiples. Si no sabes la respuesta pero intentas adivinarla, puede que tengas suerte y aciertes. Si la dejas en blanco, seguro obtienes un cero. Del mismo modo, cuando se califica a los modelos solo por su precisión, el porcentaje de preguntas que responden correctamente, se les incentiva a adivinar en lugar de decir “No lo sé”.

Para dar otro ejemplo, supongamos que a un modelo de lenguaje se le pregunta por la fecha de cumpleaños de alguien, pero no la sabe. Si adivina que es el “10 de septiembre”, tiene una posibilidad en 365 de estar en lo cierto. Decir “No lo sé” garantiza que obtendrá cero puntos. Tras miles de preguntas de prueba, el modelo de conjeturas termina teniendo mejores resultados en las tablas de puntuación que un modelo prudente que reconoce la incertidumbre.

Para las preguntas en las que hay una sola “respuesta correcta”, se pueden considerar tres categorías de respuestas: correctas, errores y abstenciones, en las que el modelo no arriesga una conjetura. Abstenerse es parte de la **humildad** , uno de los valores clave de OpenAI . La mayoría de las tablas de puntuación priorizan y clasifican los modelos en función de su precisión, pero los errores son peores que las abstenciones. Nuestra Especificación del modelo establece que es mejor indicar incertidumbre o pedir una aclaración que dar información confiada que puede ser incorrecta.

Para dar un ejemplo concreto, consideremos la evaluación SimpleQA como ejemplo de la Tarjeta del sistema de GPT‑5 .

|**Métrica**|**gpt-5-thinking-mini**|**OpenAI o4-mini**|
|---|---|---|
|**Tasa de abstención**|**52 %**|**1 %**|
|**(no se brinda una respuesta específca)**|||
|**Tasa de precisión**|**22 %**|**24 %**|
|**(respuesta correcta, más alta es mejor)**|||
|**Tasa de error**|**26 %**|**75 %**|
|**(respuesta incorrecta, más baja es mejor)**|||
|**Total**|**100 %**|**100 %**|

En términos de precisión, el modelo anterior, OpenAI o4-mini, funciona ligeramente mejor. Sin embargo, su tasa de error (es decir, la tasa de alucinaciones) es mucho más alta. Adivinar estratégicamente cuando no hay certeza mejora la precisión, pero aumenta los errores y las alucinaciones.

Al promediar los resultados de docenas de evaluaciones, la mayoría de las pruebas de referencia se basan en la métrica de precisión, pero esto implica una dicotomía falsa entre lo correcto y lo incorrecto. En evaluaciones simplistas como SimpleQA, algunos modelos alcanzan una precisión cercana al 100 % y, por lo tanto, eliminan las alucinaciones. Sin embargo, en evaluaciones más complejas y en el uso real, la precisión se limita por debajo del 100 % porque hay algunas preguntas cuya respuesta no se puede determinar por diversos motivos, como la falta de información disponible, las capacidades de razonamiento limitadas de los modelos pequeños o las ambigüedades que deben aclararse.

No obstante, las tablas de puntuación basadas únicamente en la precisión dominan los marcadores y las tarjetas de modelos, lo que motiva a los desarrolladores a crear modelos que adivinan en lugar de abstenerse. Esa es una de las razones por las que, aunque los modelos sean cada vez más avanzados, siguen generando alucinaciones, dado que proporcionan respuestas erróneas con total seguridad en lugar de reconocer la incertidumbre.

### Una mejor manera de calificar las evaluaciones

Existe una solución sencilla para esto. Penalizar los errores cometidos por confianza más que los cometidos por incertidumbre, y otorgar crédito parcial por las expresiones adecuadas de incertidumbre. Esta idea no es nueva. Desde hace tiempo, algunas pruebas estandarizadas vienen usando versiones de calificaciones negativas

para las respuestas incorrectas o crédito parcial por dejar las

preguntas en blanco para desalentar las respuestas al azar. Varios

grupos de investigación también han explorado evaluaciones que dan cuenta de la incertidumbre y la calibración.

Nuestro punto de vista es diferente. No basta con agregar algunas pruebas nuevas que tengan en cuenta la incertidumbre. Las evaluaciones basadas en la precisión, ampliamente utilizadas, deben actualizarse para que su puntuación desaliente las conjeturas. Si las principales tablas de puntuación siguen recompensando las conjeturas afortunadas, los modelos seguirán aprendiendo a hacer conjeturas. Corregir las tablas de puntuación puede ampliar la adopción de técnicas de reducción de alucinaciones, tanto las recién desarrolladas como las de investigaciones anteriores.

## Cómo se originan alucinaciones a partir de la predicción de la siguiente palabra

Ya hablamos de por qué es tan difícil deshacerse de las alucinaciones, pero ¿de dónde vienen estas imprecisiones de datos tan específicas? Después de todo, los modelos grandes preentrenados casi nunca tienen otros tipos de errores, como errores ortográficos o paréntesis mal usados. La diferencia tiene que ver con los tipos de patrones que hay en los datos.

Los modelos de lenguaje primero aprenden mediante un _preentrenamiento_ , un proceso que consiste en predecir la siguiente palabra en grandes cantidades de texto. A diferencia de lo que sucede en los problemas de aprendizaje automático tradicionales, no hay etiquetas de “verdadero/falso” asociadas a cada afirmación. El modelo solo ve ejemplos positivos de lenguaje fluido y debe aproximarse a la distribución general.

Es mucho más difícil distinguir las afirmaciones válidas de las inválidas cuando no hay ejemplos etiquetados como inválidos. Pero aún con las etiquetas, algunos errores son inevitables. Para entender el motivo, consideremos una analogía más simple. En el reconocimiento de imágenes, si se etiquetan millones de fotos de gatos y perros como “gato” o “perro”, los algoritmos pueden aprender a clasificarlas de manera confiable. Pero imagina que, en cambio, se etiquetaran las fotos de las mascotas según su fecha de nacimiento. Dado que las fechas de nacimiento son esencialmente aleatorias, esta tarea siempre produciría errores, sin importar lo avanzado que sea el algoritmo.

El mismo principio se aplica en el preentrenamiento. La ortografía y los paréntesis siguen patrones uniformes, por lo que los errores de este tipo desaparecen gradualmente. Pero los hechos arbitrarios y poco frecuentes, como el cumpleaños de una mascota, no pueden predecirse solo a partir de patrones y por ello se generan alucinaciones. Nuestro análisis explica qué tipos de alucinaciones

podrían surgir de la predicción de la siguiente palabra. Lo ideal sería

que pudieran eliminarse en etapas posteriores al preentrenamiento,

pero esto no se está logrando del todo por las razones descritas en la sección anterior.

## Conclusiones

Esperamos que el análisis estadístico de nuestro artículo de investigación aclare la naturaleza de las alucinaciones y revierta los conceptos erróneos más comunes:

- **Afirmación:** Las alucinaciones se eliminarán al mejorar la precisión porque un modelo que acierte el 100 % no alucinará nunca.

- **Hallazgo:** La precisión nunca alcanzará el 100 % porque, independientemente del tamaño del modelo y de las capacidades de búsqueda y razonamiento, algunas preguntas del mundo real son intrínsecamente imposibles de responder. **Afirmación:** Las alucinaciones son inevitables.

- **Hallazgo:** No lo son, porque los modelos de lenguaje pueden abstenerse de responder cuando no están seguros.

- **Afirmación:** Evitar las alucinaciones requiere un nivel de inteligencia que se consigue exclusivamente con modelos más grandes.

**Hallazgo:** Para un modelo pequeño puede ser _más fácil_ conocer sus propios límites. Por ejemplo, si a un modelo pequeño se le pide que responda una pregunta en maorí, si no sabe maorí, sencillamente puede responder “No lo sé”, mientras que un modelo que sabe algo de maorí tiene que determinar su nivel de confianza. Tal como se explica en el artículo, la “calibración” requiere mucho menos cálculo que la precisión.

- **Afirmación:** Las alucinaciones son una falla misteriosa en los modelos de lenguaje actuales.

**Hallazgo:** Entendemos los mecanismos estadísticos por los cuales las alucinaciones surgen y se recompensan en las evaluaciones.

- **Afirmación:** Para medir las alucinaciones, solo se necesita una buena evaluación de alucinaciones.

**Hallazgo:** Ya se publicaron evaluaciones de alucinaciones. Sin embargo, una buena evaluación de alucinaciones tiene poco efecto al lado de cientos de evaluaciones tradicionales basadas en la precisión, que penalizan la humildad y recompensan las conjeturas. En su lugar, es necesario revisar todas las métricas de evaluación principales para recompensar las expresiones de incertidumbre.

Nuestros últimos modelos tienen tasas de alucinación más bajas, y

seguimos haciendo todo lo posible para reducir aún más las tasas de errores por exceso de confianza que generan nuestros modelos de lenguaje.

<!-- Start of picture text -->
COMENTARIOS SOBRE LA TRADUCCIÓN<br>¿Qué tan fácil te resulta leer y entender esta página en español?<br>Excelente Bien<br>Deficiente<br><!-- End of picture text -->

<!-- Start of picture text -->
Alineación 2025 GPT Ética y seguridad<br><!-- End of picture text -->

#### Colaboradores del anuncio

Adam Kalai, Santosh Vempala (Georgia Tech), Ofir Nachum, Eddie Zhang, David Robinson, Saachi Jain, Eric Mitchell, Alex Beutel y Johannes Heidecke

### Sigue leyendo

Ver todos

Acelerar la ingeniería para el mundo físico

Investigación 3 nov 2026

LifeVal

Investigación 2 nov 2026

Cómo detectamos la desalineación de los modelos de vanguardia

Seguridad 29 oct 2026

Investigación Índice de las investigaciones Descripción general de la investigación Investigación económica

Últimos avances GPT-6.1 Sol GPT-6 Astra GPT-5.6 GPT-5.5

Seguridad Enfoque de seguridad

Seguridad de la implementación

Productos Business Empresa Más ChatGPT Descripción general Acerca de nosotros Historias ChatGPT Negocios Soluciones Nuestros principios Academy ChatGPT Enterprise Recursos Empleos Supply Co. ChatGPT para la educación Complementos Noticias Transmisiones en vivo Codex Historias de clientes Podcast Atención al cliente Puntos Red de socios RSS Centro de ayuda Notas de lanzamiento Comunicarse con ventas Estado de OpenAI Términos y políticas Plataforma API Desarrolladores Términos de uso Descripción general SDK de aplicaciones Política de privacidad Iniciar sesión en la API Modelos abiertos Otras políticas Documentos Documentos Recursos Foro de desarrolladores

Seguridad y privacidad Confianza y transparencia

<!-- Start of picture text -->
Español Latinoamérica<br><!-- End of picture text -->

OpenAI © 2015–2026 <u>Gestionar cookies</u>