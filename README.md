# Aplicación del paper arXiv:2604.24480 a la construcción de superficies de volatilidad implícita

Este repositorio desarrolla una aplicación práctica del resultado presentado por
Wolfgang Schadner en `An Explicit Solution to Black-Scholes Implied Volatility`
arXiv:2604.24480v1. El paper propone una forma explícita de invertir la fórmula de
Black-Scholes y obtener la volatilidad implícita mediante el cuantil de una distribución
Gaussiana Inversa. La motivación de este proyecto es llevar ese resultado desde su
formulación matemática a un caso empírico: calcular volatilidades implícitas con datos
reales de mercado y organizar esas volatilidades en una superficie de volatilidad.

La aplicación descarga cadenas de opciones desde Yahoo Finance, transforma los precios
observados en volatilidades implícitas usando la fórmula explícita del paper y representa
la estructura resultante en varios formatos: superficie 3D, mapa de calor, sonrisas por
vencimiento, estructura temporal e índice de skew.

El objetivo no es sustituir una plataforma profesional de valoración, sino mostrar de
forma transparente cómo un resultado teórico reciente puede aplicarse al análisis de una
cadena de opciones real. La aplicación mantiene visible el flujo completo:

```text
Precio de opción
    -> paridad put-call
    -> normalización Black-Scholes
    -> fórmula explícita con cuantil de Gaussiana Inversa
    -> IV por contrato
    -> nube de puntos
    -> superficie interpolada
```

## Motivación

El punto de partida del proyecto es una cuestión clásica en finanzas cuantitativas:
Black-Scholes proporciona el precio de una opción si se conoce la volatilidad, pero en
mercado ocurre lo contrario. Lo observable es el precio de la opción; la volatilidad debe
inferirse.

Formalmente, dada una opción con precio de mercado `C_mercado`, se busca la volatilidad
`sigma` que satisface:

```math
C_{BS}(\sigma) = C_{mercado}
```

Esa `sigma` es la volatilidad implícita. Durante décadas, esta inversión se ha tratado
como un problema numérico. En la práctica, se resolvía mediante algoritmos iterativos:
Newton-Raphson, bisección, métodos de Householder o aproximaciones racionales.

El interés del paper de Schadner es que reformula el problema. En lugar de buscar la raíz
de una ecuación no lineal, identifica una relación exacta entre el precio Black-Scholes
normalizado y la función de distribución de una Gaussiana Inversa. Al invertir esa CDF,
la volatilidad queda expresada directamente mediante una función cuantil.

La pregunta natural que motiva esta aplicación es:

> Si el paper permite obtener la IV contrato a contrato sin iteraciones, ¿cómo se ve la
> superficie de volatilidad resultante cuando aplicamos la fórmula a una cadena real de
> opciones?

La aplicación responde a esa pregunta. Para cada opción válida, calcula una IV explícita.
Después, al repetir el procedimiento sobre muchos strikes y vencimientos, se obtiene una
nube de puntos que aproxima la superficie empírica de volatilidad.

## Por qué estudiar la superficie de volatilidad

En el modelo Black-Scholes clásico, la volatilidad es constante. Bajo ese supuesto, todas
las opciones sobre el mismo subyacente deberían compartir la misma volatilidad,
independientemente del strike y del vencimiento:

```math
\sigma_{imp}(K,T)=\sigma_0
```

En ese caso, la superficie de volatilidad sería un plano horizontal.

Sin embargo, los mercados reales no se comportan así. La volatilidad implícita observada
depende sistemáticamente del strike y del plazo:

```math
\sigma_{imp}=\sigma_{imp}(K,T)
```

o, en la parametrización usada por esta aplicación:

```math
\sigma_{imp}=\sigma_{imp}(\ln(K/F),T)
```

La superficie de volatilidad resume, en un único objeto, cómo el mercado valora el riesgo
de movimientos futuros para distintos escenarios y horizontes temporales. No es
solamente un gráfico: es una representación de la estructura de precios de las opciones.

La superficie permite estudiar:

- El nivel general de volatilidad implícita.
- La estructura temporal de la volatilidad.
- El skew por strike.
- La prima de protección en puts OTM.
- La diferencia entre riesgo de corto y largo plazo.
- La presencia de tensiones o eventos concentrados en determinados vencimientos.
- La forma de la sonrisa de volatilidad en distintos mercados.

En índices de renta variable, por ejemplo, suele observarse que los puts OTM tienen mayor
IV que las opciones ATM. Esto genera una superficie inclinada hacia el lado izquierdo,
asociada a la demanda estructural de protección frente a caídas severas.

## Qué problema resuelve este proyecto

El proyecto conecta tres niveles que normalmente aparecen separados:

1. **Resultado teórico del paper:** inversión explícita de Black-Scholes mediante la
   distribución Gaussiana Inversa.
2. **Implementación computacional:** traducción de la fórmula a Python usando
   `scipy.stats.invgauss.ppf`.
3. **Aplicación empírica:** construcción de una superficie de volatilidad con datos
   observados de Yahoo Finance.

El resultado es una herramienta que muestra cómo el avance matemático del paper puede
incorporarse a un flujo práctico de análisis de opciones:

```text
Cadena de opciones
    -> filtrado de contratos
    -> precio mid
    -> forward y descuento
    -> log-moneyness
    -> paridad put-call
    -> fórmula explícita de IV
    -> Greeks
    -> superficie, heatmap, smiles, skew y estructura temporal
```

## Características principales

- Descarga de datos de opciones con `yfinance`.
- Cálculo de volatilidad implícita sin Newton-Raphson.
- Uso de puts OTM como instrumentos principales para analizar skew de renta variable.
- Conversión de puts a calls sintéticas mediante paridad put-call.
- Representación 3D de la superficie de volatilidad.
- Mapa de calor de la misma superficie.
- Curvas de sonrisa/smirk por vencimiento.
- Estructura temporal de IV ATM, 25-delta y 10-delta.
- Índice de skew por vencimiento.
- Tabla de contratos con IV, delta, gamma, vega, bid, ask, volumen y open interest.
- Soporte para tickers de renta variable, ETFs, commodities y proxies FX listados en Yahoo Finance.
- Documentación dentro de la propia aplicación con fórmulas y explicación metodológica.

## Arquitectura de la aplicación

La aplicación principal está contenida en `streamlit_app.py`. El código está organizado
en bloques funcionales:

| Bloque | Función |
|---|---|
| Configuración de página | Define layout, tema visual y estilos CSS. |
| Fórmula cerrada IV | Implementa la inversión explícita mediante `invgauss.ppf`. |
| Descarga de datos | Obtiene spot, vencimientos y cadenas de opciones con `yfinance`. |
| Limpieza de contratos | Filtra precios inválidos, vencimientos no deseados y contratos ilíquidos. |
| Cálculo de IV y Greeks | Calcula `iv`, `delta`, `gamma` y `vega` para cada opción. |
| Estructura temporal | Interpola ATM, 25-delta y 10-delta por vencimiento. |
| Figuras Plotly | Construye superficie 3D, heatmap, smiles, skew y term structure. |
| Interfaz Streamlit | Organiza la app en pestañas, métricas, filtros y explicaciones. |

Funciones principales:

| Función | Descripción |
|---|---|
| `iv_closed_form(C, K, F, D, T)` | Calcula IV explícita de una call. |
| `iv_from_put(P, K, F, D, T)` | Convierte un put en call sintética y calcula IV. |
| `fetch_options_data(ticker, risk_free_rate, max_T)` | Descarga, filtra y transforma la cadena de opciones. |
| `compute_term_structure(df)` | Construye ATM IV, 25-delta IV, 10-delta IV y skew por vencimiento. |
| `fig_3d_surface(df)` | Dibuja nube de puntos e interpolación 3D. |
| `fig_iv_heatmap(df)` | Dibuja la superficie vista desde arriba. |
| `fig_iv_smile_moneyness(df, selected)` | Dibuja smiles en log-moneyness. |
| `fig_iv_smile_delta(df, selected)` | Dibuja smiles en espacio delta. |
| `fig_term_structure(term_df)` | Dibuja estructura temporal. |
| `fig_skew_index(term_df)` | Dibuja índice de skew por vencimiento. |

## Estructura del repositorio

```text
.
├── streamlit_app.py               # Aplicación principal de Streamlit
├── streamlit_app_explicado.md     # Explicación técnica detallada del código
├── vol_surface.py                 # Script auxiliar para superficie de volatilidad
├── vol_surface_explained.md       # Explicación previa de la metodología
├── volatilidad_implicita.tex      # Apuntes teóricos en LaTeX
├── spy_iv_dashboard.png           # Ejemplo de salida visual
└── README.md                      # Este documento
```

## Instalación

Se recomienda usar un entorno virtual.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install streamlit numpy pandas scipy plotly yfinance
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install streamlit numpy pandas scipy plotly yfinance
```

## Ejecución

Desde la raíz del proyecto:

```bash
streamlit run streamlit_app.py
```

Streamlit abrirá una URL local, normalmente:

```text
http://localhost:8501
```

## Parámetros de la aplicación

La barra lateral permite configurar:

| Parámetro | Descripción |
|---|---|
| Activo rápido | Selector de tickers predefinidos. |
| Ticker manual | Campo para introducir cualquier ticker compatible con Yahoo Finance. |
| Tasa libre de riesgo | Tasa continua usada para calcular descuento y forward. |
| Máximo vencimiento | Horizonte máximo de expiración incluido en la muestra. |

Tickers recomendados:

| Ticker | Descripción |
|---|---|
| `SPY` | ETF del S&P 500, muy líquido. |
| `QQQ` | ETF del Nasdaq 100. |
| `IWM` | ETF del Russell 2000. |
| `GLD` | ETF de oro. |
| `TLT` | ETF de bonos de largo plazo. |
| `FXE` | ETF proxy EUR/USD. |
| `FXY` | ETF proxy JPY/USD. |
| `FXB` | ETF proxy GBP/USD. |
| `FXA` | ETF proxy AUD/USD. |
| `UUP` | ETF del índice dólar estadounidense. |

Yahoo Finance normalmente no proporciona cadenas gratuitas de opciones OTC sobre pares
spot como `EURUSD=X`. Por eso, para FX se usan ETFs de divisas con opciones listadas.

## Concepto de superficie de volatilidad

La superficie de volatilidad es la función que asigna una volatilidad implícita a cada
combinación de strike y vencimiento:

```math
(K,T) \mapsto \sigma_{imp}(K,T)
```

En la práctica, no se suele trabajar directamente con el strike absoluto `K`, porque un
strike de 500 tiene significados distintos para activos distintos o incluso para el
mismo activo en vencimientos diferentes. Por eso se usa una coordenada relativa al
forward:

```math
k=\ln(K/F)
```

La superficie se estudia entonces como:

```math
(k,T) \mapsto \sigma_{imp}(k,T)
```

Esta elección tiene ventajas importantes:

- `k = 0` representa el punto at-the-forward.
- `k < 0` representa strikes por debajo del forward.
- `k > 0` representa strikes por encima del forward.
- El logaritmo convierte ratios en distancias aditivas.
- La coordenada coincide con la forma natural de Black-Scholes.

### Interpretación de los ejes

| Eje | Variable | Interpretación |
|---|---|---|
| X | `log-moneyness = ln(K/F)` | Posición relativa del strike frente al forward. |
| Y | `T` | Tiempo al vencimiento en años. |
| Z | `IV` | Volatilidad implícita anualizada. |

Un punto de la superficie representa una opción concreta:

```math
\left(\ln(K_i/F_i), T_i, \sigma_i\right)
```

Si `k = -10%`, el strike está aproximadamente un 10% por debajo del forward. En un put,
eso corresponde a una opción fuera del dinero. Si `k = -50%`, el put representa una
protección de cola mucho más extrema, pero sus datos suelen ser menos líquidos.

### Smile, smirk y skew

Un corte de la superficie para un vencimiento fijo produce una curva de volatilidad por
strike:

```math
k \mapsto \sigma_{imp}(k,T_0)
```

Esta curva se denomina smile o smirk de volatilidad.

En un smile simétrico, la IV aumenta en ambos extremos. Esta forma puede aparecer en
mercados donde los movimientos extremos al alza y a la baja se valoran de forma más
equilibrada.

En un smirk de renta variable, la IV suele ser mayor en el lado izquierdo:

```math
\sigma_{imp}(k<0,T) > \sigma_{imp}(0,T)
```

Esto ocurre porque los puts OTM incorporan una prima de protección frente a caídas
severas. El skew resume esa pendiente por strike.

Una forma habitual de medirlo es:

```math
Skew(T)=\sigma_{25\Delta}(T)-\sigma_{ATM}(T)
```

Un skew positivo indica que el mercado asigna mayor IV a puts OTM que a opciones ATM
del mismo vencimiento.

### Estructura temporal

Además del strike, la IV varía con el vencimiento:

```math
T \mapsto \sigma_{imp}(k_0,T)
```

La estructura temporal permite analizar si el mercado asigna más prima de volatilidad al
corto o al largo plazo.

| Forma | Interpretación |
|---|---|
| Contango | La IV aumenta con el plazo. |
| Backwardation | La IV corta supera a la IV larga. |
| Curva plana | El mercado no diferencia mucho entre plazos. |

Una backwardation marcada suele asociarse a tensión de corto plazo, aunque su lectura
debe contrastarse con liquidez, calendario de eventos y calidad de datos.

## Metodología financiera

### 1. Inputs observados

Para cada contrato de opción se toman:

| Variable | Descripción |
|---|---|
| `S` | Precio spot del subyacente. |
| `K` | Strike de la opción. |
| `T` | Tiempo al vencimiento en años. |
| `r` | Tasa libre de riesgo continua. |
| `P` | Precio del put, preferentemente mid-price. |

El precio usado para cada put es:

```math
P_{mid} = \frac{bid + ask}{2}
```

si `bid` y `ask` son positivos. Si no, se usa `lastPrice` cuando está disponible.

### 2. Descuento y forward

Para cada vencimiento se calcula:

```math
D = e^{-rT}
```

```math
F = \frac{S}{D} = S e^{rT}
```

En esta versión no se modela explícitamente dividend yield. Para índices o ETFs con
dividendos, una extensión natural sería:

```math
F = S e^{(r-q)T}
```

donde `q` es el dividend yield continuo.

### 3. Log-moneyness

La superficie se representa frente a:

```math
k = \ln\left(\frac{K}{F}\right)
```

Esta coordenada tiene varias ventajas:

- Coloca el punto at-the-forward en `k = 0`.
- Permite comparar strikes de forma relativa al forward.
- Es la variable natural en la formulación lognormal de Black-Scholes.
- Es la variable usada en la fórmula explícita del paper.

Interpretación:

| Condición | Log-moneyness | Interpretación para puts |
|---|---:|---|
| `K = F` | `k = 0` | At-the-forward. |
| `K < F` | `k < 0` | Put OTM. |
| `K > F` | `k > 0` | Put ITM. |

Ejemplo:

```math
F = 600,\quad K = 540
```

```math
\frac{K}{F}=0.90,\qquad k=\ln(0.90)\approx -0.105
```

En el gráfico aparece aproximadamente como `-10.5%`.

### 4. Paridad put-call

La fórmula explícita se aplica sobre calls. Como la aplicación usa principalmente puts,
cada put se convierte en una call sintética mediante paridad put-call:

```math
C - P = D(F-K)
```

Por tanto:

```math
C = P + D(F-K)
```

En el código:

```python
def iv_from_put(P: float, K: float, F: float, D: float, T: float) -> float:
    C = P + D * (F - K)
    return iv_closed_form(C, K, F, D, T) if C > 0 else np.nan
```

### 5. Normalización Black-Scholes

El precio de la call se normaliza como:

```math
c = \frac{C}{DF}
```

Además se define la volatilidad total:

```math
v = \sigma\sqrt{T}
```

donde `sigma` es la volatilidad implícita anualizada.

## Fórmula explícita de volatilidad implícita

Para `k != 0`, el artículo de Schadner establece una identidad entre el precio
Black-Scholes normalizado y la CDF de una distribución Gaussiana Inversa:

```math
\frac{1-c}{m}
=
\mathcal{F}_{IG}
\left(
    \frac{4}{v^2};
    \frac{2}{|k|},
    1
\right)
```

donde:

```math
m = \min(1,e^k)
```

Al invertir la CDF:

```math
x^*
=
\mathcal{F}_{IG}^{-1}
\left(
    \frac{1-c}{m};
    \frac{2}{|k|},
    1
\right)
```

y como:

```math
x^* = \frac{4}{v^2}
```

se obtiene:

```math
v = \frac{2}{\sqrt{x^*}}
```

Por tanto:

```math
\sigma
=
\frac{2}{\sqrt{T}}
\left[
\mathcal{F}_{IG}^{-1}
\left(
    \frac{1-c}{m};
    \frac{2}{|k|},
    1
\right)
\right]^{-1/2}
```

En Python:

```python
prob = np.clip((1.0 - c) / m, 1e-10, 1 - 1e-10)
x = invgauss.ppf(prob, mu=2.0 / abs(k))
sigma = (2.0 / np.sqrt(x)) / np.sqrt(T)
```

### Caso ATM

Cuando `k` está muy cerca de cero, se usa el límite at-the-money:

```math
\sigma_{ATM}
=
\frac{2}{\sqrt{T}}
\Phi^{-1}
\left(
    \frac{c+1}{2}
\right)
```

En el código:

```python
arg = np.clip((c + 1.0) / 2.0, 1e-10, 1 - 1e-10)
v = 2.0 * norm.ppf(arg)
sigma = v / np.sqrt(T)
```

## Construcción de la superficie

Cada contrato válido produce un punto:

```math
\left(
    100\ln(K_i/F_i),
    T_i,
    100\sigma_i
\right)
```

En código:

```python
xs = sd["log_moneyness"].values * 100
ys = sd["T"].values
zs = sd["iv_pct"].values
```

La nube de puntos se representa directamente con `Scatter3d`. Estos puntos son las
observaciones calculadas contrato por contrato.

Para dibujar una lámina continua se usa interpolación lineal:

```python
xi = np.linspace(xs.min(), xs.max(), 60)
yi = np.linspace(ys.min(), ys.max(), 50)
XI, YI = np.meshgrid(xi, yi)
ZI = griddata((xs, ys), zs, (XI, YI), method="linear")
```

La interpolación no es una calibración financiera de superficie. No impone ausencia de
arbitraje y no equivale a un modelo SVI, SABR o Dupire. Es una herramienta visual para
interpretar la geometría de los puntos observados.

## Interpretación del skew

En índices de renta variable, lo habitual es observar mayor IV en strikes por debajo
del forward:

```math
\sigma_{imp}(k=-50\%)
>
\sigma_{imp}(k=-10\%)
>
\sigma_{imp}(k=0)
```

Esto refleja una prima por protección de cola izquierda. Los puts OTM profundos suelen
tener IV superior porque los inversores demandan cobertura frente a caídas severas.

Sin embargo, los extremos de la cadena pueden ser ruidosos. En strikes muy alejados
pueden aparecer:

- bajo volumen,
- bid/ask amplio,
- precios antiguos,
- open interest bajo,
- interpolación inestable por escasez de puntos.

Por eso, la lectura robusta del skew suele basarse en referencias como ATM, 25-delta y
10-delta, no únicamente en strikes extremos.

## Estructura temporal

La aplicación calcula, por vencimiento:

```math
\sigma_{ATM}(T)
```

```math
\sigma_{25\Delta}(T)
```

```math
\sigma_{10\Delta}(T)
```

y el índice de skew:

```math
Skew(T)=\sigma_{25\Delta}(T)-\sigma_{ATM}(T)
```

Interpretación general:

| Forma | Lectura |
|---|---|
| Contango | La IV aumenta con el vencimiento. |
| Backwardation | La IV corta supera a la IV larga. |
| Curva plana | El mercado no diferencia de forma marcada el riesgo por plazo. |
| Skew elevado | Los puts OTM incorporan una prima significativa frente al ATM. |

## Cálculo de Greeks

Una vez obtenida la IV, se calculan delta, gamma y vega con fórmulas Black-Scholes:

```math
d_1 =
\frac{\ln(F/K)}{\sigma\sqrt{T}}
+ \frac{\sigma\sqrt{T}}{2}
```

Para puts:

```math
\Delta_{put} = \Phi(d_1)-1
```

Gamma:

```math
\Gamma =
\frac{\phi(d_1)}{F\sigma\sqrt{T}}
```

Vega:

```math
\mathcal{V}
=
FD\phi(d_1)\sqrt{T}
```

Estas magnitudes se muestran en la tabla de cadena de opciones.

## Limitaciones

### Datos

- `yfinance` no garantiza datos en tiempo real.
- Los precios de Yahoo Finance pueden estar retrasados o incompletos.
- Las opciones ilíquidas pueden tener bid/ask amplio o precios desactualizados.
- No hay datos gratuitos de opciones OTC de FX spot; se usan ETFs proxy.

### Modelo

- Se usa Black-Scholes europeo.
- No se modelan dividendos explícitos.
- No se ajusta una superficie libre de arbitraje.
- No se estima volatilidad local ni densidad neutral al riesgo.
- La interpolación `griddata` es visual, no un modelo de valoración.

### Interpretación

Los gráficos no constituyen recomendación financiera. La aplicación es una herramienta
educativa y exploratoria para estudiar superficies de volatilidad implícita.

## Posibles mejoras

- Incorporar dividend yield o forwards observados.
- Filtrar contratos por spread relativo:

```math
\frac{ask-bid}{mid} < \epsilon
```

- Añadir calibración SVI por vencimiento.
- Añadir SABR para activos de tipos/FX.
- Guardar snapshots históricos de superficies.
- Comparar la fórmula explícita con Newton-Raphson o Jäckel.
- Añadir chequeos de arbitraje estático.
- Integrar Interactive Brokers u otra fuente profesional de datos.

## Referencias

1. Wolfgang Schadner, `An Explicit Solution to Black-Scholes Implied Volatility`,
   arXiv:2604.24480v1, 2026.  
   https://arxiv.org/abs/2604.24480

2. Fischer Black and Myron Scholes, `The Pricing of Options and Corporate Liabilities`,
   Journal of Political Economy, 81, 1973, DOI: 10.1086/260062.  
   https://doi.org/10.1086/260062

3. Robert C. Merton, `Theory of Rational Option Pricing`, Bell Journal of Economics and
   Management Science, 1973.

4. Peter Jäckel, `Let's Be Rational`, Wilmott, 2015, DOI: 10.1002/wilm.10395.  
   https://doi.org/10.1002/wilm.10395

5. Jim Gatheral, `The Volatility Surface: A Practitioner's Guide`, Wiley, 2006.

6. John C. Hull, `Options, Futures, and Other Derivatives`, Pearson.

7. SciPy documentation, `scipy.stats.invgauss`.  
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgauss.html

8. SciPy documentation, `scipy.interpolate.griddata`.  
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

9. yfinance documentation and project repository.  
   https://github.com/ranaroussi/yfinance

## Licencia y uso

Este repositorio se proporciona con fines educativos y de investigación. Antes de usarlo
con fines profesionales o de trading, se recomienda validar los datos, revisar la
microestructura de mercado, incorporar dividendos/forwards observados y aplicar filtros
de liquidez más estrictos.
