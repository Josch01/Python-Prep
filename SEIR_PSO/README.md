# Ajuste de modelo SEIR con PSO

Este ejemplo muestra cómo utilizar un algoritmo de *Particle Swarm Optimization* (PSO) para ajustar los parámetros de un modelo epidemiológico SEIR con una tasa de contagio `beta(t)` oscilatoria. 

## Uso

```
python seir_pso.py data/infected_example.csv
```

El archivo de entrada debe contener dos columnas: la primera con el tiempo (días o semanas) y la segunda con el número de infectados. Los datos se normalizan automáticamente antes de realizar el ajuste.

El script ejecuta el PSO por intervalos y luego realiza un ajuste global sobre todo el conjunto de datos para obtener los parámetros finales.

Al terminar, se muestra una gráfica con los puntos de infectados observados y la curva ajustada en rojo.
