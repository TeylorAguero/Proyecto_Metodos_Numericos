# =============================================================================
# SPLINE CÚBICO NATURAL
# =============================================================================
# Estructura:
#   SECCIÓN 1 — LÓGICA MATEMÁTICA  (sin dependencias visuales)
#   SECCIÓN 2 — INTERFAZ GRÁFICA   (tkinter + matplotlib)
# =============================================================================


import subprocess
import sys
import math


# =============================================================================
# SECCIÓN 1 — LÓGICA MATEMÁTICA
# =============================================================================
# Contiene todo el núcleo numérico del método.
# No depende de tkinter ni matplotlib; puede usarse de forma independiente.
# =============================================================================


# -----------------------------------------------------------------------------
# 1.1  EVALUADOR DE EXPRESIONES
# -----------------------------------------------------------------------------
# Permite que el usuario escriba expresiones como "sin(pi/2)" o "2^3"
# en los campos de texto, en lugar de sólo números literales.
#
# Se restringe el entorno de eval() a un diccionario de funciones seguras
# (permitidas) para evitar ejecución de código arbitrario.
# -----------------------------------------------------------------------------

permitidas = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "ln": math.log,
    "log": math.log10,
    "pi": math.pi,
    "e": math.e,
    "abs": abs
}

def evaluar(expr):
    """Convierte una cadena de texto en un número flotante.
    Acepta funciones matemáticas básicas y el operador ^ como potencia."""
    expr = expr.strip().lower()
    expr = expr.replace("^", "**")
    return eval(expr, {"__builtins__": {}}, permitidas)


# -----------------------------------------------------------------------------
# 1.2  RESOLUCIÓN DE SISTEMA LINEAL — ELIMINACIÓN DE GAUSS CON PIVOTEO PARCIAL
# -----------------------------------------------------------------------------
# Resuelve el sistema A·x = b que se forma al plantear las condiciones
# de continuidad del spline en los nodos interiores.
#
# Algoritmo:
#   1. Pivoteo parcial: busca la fila con el mayor valor absoluto en la
#      columna actual y la intercambia con la fila pivote.  Esto mejora la
#      estabilidad numérica.
#   2. Eliminación hacia adelante: convierte A en una matriz triangular
#      superior aplicando operaciones de fila.
#   3. Sustitución hacia atrás: resuelve el sistema desde la última
#      incógnita hasta la primera.
#
# Devuelve None si la diagonal contiene un cero (sistema sin solución única).
# -----------------------------------------------------------------------------

def resolver_sistema_gauss(A, b):
    """Resuelve A·x = b con eliminación gaussiana y pivoteo parcial.
    Retorna la lista de soluciones x, o None si el sistema es singular."""
    n = len(b)

    for i in range(n):
        # --- pivoteo parcial ---
        mayor = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[mayor][i]):
                mayor = k
        A[i], A[mayor] = A[mayor], A[i]
        b[i], b[mayor] = b[mayor], b[i]

        if A[i][i] == 0:
            return None  # Sistema singular

        # --- eliminación hacia adelante ---
        for k in range(i + 1, n):
            factor = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
            b[k] -= factor * b[i]

    # --- sustitución hacia atrás ---
    x = [0] * n
    for i in range(n - 1, -1, -1):
        suma = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - suma) / A[i][i]

    return x


# -----------------------------------------------------------------------------
# 1.3  CONSTRUCCIÓN Y EVALUACIÓN DEL SPLINE CÚBICO NATURAL
# -----------------------------------------------------------------------------
# Un spline cúbico natural conecta n puntos con n-1 polinomios cúbicos.
# Las condiciones que se imponen son:
#   • Cada tramo pasa exactamente por sus dos nodos extremos.
#   • La primera y segunda derivada son continuas en los nodos interiores.
#   • Condición "natural": segunda derivada = 0 en los extremos (M₀ = Mₙ = 0).
#
# Para encontrar los momentos Mᵢ (= S''(xᵢ)) se arma el sistema tridiagonal
# y se resuelve con Gauss.  Luego la fórmula de cada tramo es:
#
#   S(x) = M₁(x₂-x)³/6h + M₂(x-x₁)³/6h
#         + (y₁ - M₁h²/6)(x₂-x)/h
#         + (y₂ - M₂h²/6)(x-x₁)/h
# -----------------------------------------------------------------------------

def spline_natural(x, y, valor):
    """Calcula el spline cúbico natural y evalúa S(valor).

    Parámetros
    ----------
    x, y : listas de puntos conocidos (deben tener al menos 3 elementos)
    valor: punto donde se desea interpolar

    Retorna
    -------
    resultado : S(valor)
    M         : lista de momentos Mᵢ
    h         : lista de distancias entre nodos
    datos     : diccionario con toda la información intermedia
    err       : cadena de error o None
    """
    n = len(x)

    if n < 3:
        return None, None, None, None, "Se necesitan al menos 3 puntos."

    # --- paso 1: calcular distancias hᵢ = xᵢ₊₁ - xᵢ ---
    h = []
    for i in range(n - 1):
        hi = x[i + 1] - x[i]
        if hi == 0:
            return None, None, None, None, "No puede haber x repetidos."
        h.append(hi)

    # --- paso 2: armar sistema tridiagonal para los momentos ---
    # La primera y última fila imponen M₀ = 0 y Mₙ = 0 (condición natural).
    # Las filas interiores expresan la continuidad de la segunda derivada.
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n

    A[0][0]         = 1          # M₀ = 0
    A[n - 1][n - 1] = 1          # Mₙ = 0

    for i in range(1, n - 1):
        A[i][i - 1] = h[i - 1]
        A[i][i]     = 2 * (h[i - 1] + h[i])
        A[i][i + 1] = h[i]
        b[i] = 6 * (
            (y[i + 1] - y[i]) / h[i]
            - (y[i] - y[i - 1]) / h[i - 1]
        )

    # --- paso 3: resolver el sistema para obtener los momentos ---
    M = resolver_sistema_gauss(A, b)
    if M is None:
        return None, None, None, None, "No se pudo resolver."

    # --- paso 4: localizar el intervalo que contiene a 'valor' ---
    pos = -1
    for i in range(n - 1):
        if x[i] <= valor <= x[i + 1]:
            pos = i
            break

    if pos == -1:
        return None, None, None, None, "Valor fuera del rango."

    # --- paso 5: evaluar la fórmula del spline en el intervalo encontrado ---
    i  = pos
    x1, x2 = x[i], x[i + 1]
    y1, y2 = y[i], y[i + 1]
    hi      = h[i]
    M1, M2  = M[i], M[i + 1]

    resultado = (
        M1 * (x2 - valor) ** 3 / (6 * hi)
        + M2 * (valor - x1) ** 3 / (6 * hi)
        + (y1 - M1 * hi ** 2 / 6) * (x2 - valor) / hi
        + (y2 - M2 * hi ** 2 / 6) * (valor - x1) / hi
    )

    datos = {
        "x": x, "y": y, "h": h, "M": M,
        "x1": x1, "x2": x2, "y1": y1, "y2": y2,
        "hi": hi, "valor": valor, "resultado": resultado,
        "M1": M1, "M2": M2
    }

    return resultado, M, h, datos, None


def evaluar_spline(x_pts, y_pts, M, h, t):
    """Evalúa el spline en un punto arbitrario t.
    Busca el intervalo correcto y aplica la fórmula cúbica.
    Retorna None si t está fuera del rango."""
    for i in range(len(x_pts) - 1):
        if x_pts[i] <= t <= x_pts[i + 1]:
            x1, x2 = x_pts[i], x_pts[i + 1]
            y1, y2 = y_pts[i], y_pts[i + 1]
            hi      = h[i]
            M1, M2  = M[i], M[i + 1]
            return (
                M1 * (x2 - t) ** 3 / (6 * hi)
                + M2 * (t - x1) ** 3 / (6 * hi)
                + (y1 - M1 * hi ** 2 / 6) * (x2 - t) / hi
                + (y2 - M2 * hi ** 2 / 6) * (t - x1) / hi
            )
    return None


# =============================================================================
# SECCIÓN 2 — INTERFAZ GRÁFICA
# =============================================================================
# Depende de tkinter (ventanas) y matplotlib (gráficas).
# Se instalan automáticamente si no están presentes.
# Todo lo de aquí en adelante es exclusivamente visual/interactivo.
# =============================================================================


# -----------------------------------------------------------------------------
# 2.0  INSTALACIÓN AUTOMÁTICA DE DEPENDENCIAS VISUALES
# -----------------------------------------------------------------------------

for pkg in ["matplotlib", "numpy"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Instalando {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import rcParams
import numpy as np


# -----------------------------------------------------------------------------
# 2.1  PALETA DE COLORES Y CONFIGURACIÓN GLOBAL DE MATPLOTLIB
# -----------------------------------------------------------------------------
# Se definen todos los colores como constantes para mantener consistencia
# y facilitar cambios de tema desde un único lugar.
# rcParams aplica el tema oscuro a todas las figuras de matplotlib.
# -----------------------------------------------------------------------------

BG      = "#070711"
PANEL   = "#0d0d18"
CARD    = "#161628"
CARD2   = "#1d1d35"
ACCENT  = "#6c63ff"
ACCENT2 = "#ff4d94"
TEXT    = "#f1f1ff"
SUBTEXT = "#9696c7"
SUCCESS = "#00e59b"
BORDER  = "#2d2d4f"

# Colores exclusivos para la gráfica de momentos Mᵢ
MI_BG     = "#0f1020"
MI_GRID   = "#1a1a30"
MI_ZERO   = "#555577"
MI_SPINE  = "#333355"
MI_ZERO_C = "#9c95ff"   # etiqueta cuando M = 0
MI_INT_C  = "#ff80b2"   # etiqueta cuando M ≠ 0

rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   "#0f1020",
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  SUBTEXT,
    "xtick.color":      SUBTEXT,
    "ytick.color":      SUBTEXT,
    "text.color":       TEXT,
    "grid.color":       "#2a2a46",
    "grid.alpha":       0.2,
    "legend.facecolor": CARD,
    "legend.edgecolor": BORDER,
    "legend.labelcolor": TEXT
})


# -----------------------------------------------------------------------------
# 2.2  VENTANA DE EXPLICACIÓN TEÓRICA
# -----------------------------------------------------------------------------
# Abre una ventana secundaria (Toplevel) con tarjetas scrolleables que
# explican los conceptos matemáticos detrás del método.
# No realiza ningún cálculo; es puramente informativa.
# -----------------------------------------------------------------------------

def mostrar_explicacion():
    ventana_exp = tk.Toplevel(ventana)
    ventana_exp.title("Explicación del Procedimiento")
    ventana_exp.geometry("1200x750")
    ventana_exp.configure(bg=BG)

    tk.Label(
        ventana_exp,
        text="Explicación Completa del Método",
        bg=BG, fg=ACCENT,
        font=("Segoe UI", 24, "bold")
    ).pack(pady=(20, 10))

    frame_principal = tk.Frame(ventana_exp, bg=BG)
    frame_principal.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    canvas_exp = tk.Canvas(frame_principal, bg=BG, highlightthickness=0)
    scrollbar  = tk.Scrollbar(frame_principal, orient="vertical", command=canvas_exp.yview)
    canvas_exp.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas_exp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    contenido      = tk.Frame(canvas_exp, bg=BG)
    ventana_canvas = canvas_exp.create_window((0, 0), window=contenido, anchor="nw")

    canvas_exp.bind("<Configure>",
                    lambda e: canvas_exp.itemconfig(ventana_canvas, width=e.width))
    contenido.bind("<Configure>",
                   lambda e: canvas_exp.configure(scrollregion=canvas_exp.bbox("all")))
    canvas_exp.bind_all("<MouseWheel>",
                        lambda e: canvas_exp.yview_scroll(int(-1*(e.delta/120)), "units"))

    def tarjeta(titulo, texto, color):
        """Crea una tarjeta visual con encabezado de color y cuerpo de texto."""
        card = tk.Frame(contenido, bg=CARD2,
                        highlightbackground=color, highlightthickness=1)
        card.pack(fill=tk.X, expand=True, padx=20, pady=10)

        header = tk.Frame(card, bg=color, height=40)
        header.pack(fill=tk.X)
        tk.Label(header, text="  " + titulo, bg=color, fg="white",
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=8)

        body = tk.Frame(card, bg=CARD2, padx=25, pady=20)
        body.pack(fill=tk.BOTH, expand=True)
        tk.Label(body, text=texto, bg=CARD2, fg=TEXT,
                 justify="left", anchor="w", wraplength=1000,
                 font=("Consolas", 11)).pack(anchor="w", fill=tk.X)

    tarjeta("📘 INTRODUCCIÓN", """
Un spline cúbico natural es un método de interpolación.

La interpolación se utiliza cuando conocemos algunos puntos
y queremos estimar valores intermedios.

El spline construye una curva suave que pasa exactamente
por todos los puntos dados.

Cada tramo entre puntos utiliza un polinomio cúbico.

La ventaja principal del spline es que la curva se ve natural
y suave, sin cambios bruscos.
""", "#4d8cff")

    tarjeta("△ DISTANCIAS h", """
Los valores h representan la distancia horizontal
entre dos valores consecutivos de x.

La fórmula es:

h = x(i+1) - x(i)

Estos valores ayudan a construir el sistema de ecuaciones.
""", "#9c88ff")

    tarjeta("◇ MOMENTOS Mi", """
Los momentos Mi representan las segundas derivadas
del spline en cada nodo.

Estos valores indican la curvatura.

Si Mi es grande:
la curva se dobla más.

Si Mi es pequeño:
la curva es más suave.

En el spline natural:

M0 = 0
Mn = 0

Esto significa que los extremos tienen curvatura cero.
""", ACCENT2)

    tarjeta("🧮 MÉTODO DE GAUSS", """
Para obtener los momentos Mi,
el programa construye un sistema de ecuaciones.

Ese sistema se resuelve usando el método de eliminación de Gauss.

Pasos generales:

1. Se organiza la matriz.
2. Se eliminan valores debajo de la diagonal.
3. Se obtiene una matriz triangular.
4. Se resuelve desde abajo hacia arriba.
""", "#00cfff")

    tarjeta("📐 FÓRMULA DEL SPLINE", """
La fórmula utilizada es:

S(x) =

M1(x2-x)^3 / 6h

+

M2(x-x1)^3 / 6h

+

(y1-M1h²/6)(x2-x)/h

+

(y2-M2h²/6)(x-x1)/h

Donde:

x1 y x2:
extremos del intervalo.

M1 y M2:
momentos del intervalo.

h:
distancia entre puntos.

x:
valor que queremos interpolar.
""", "#00e59b")

    tarjeta("📚 FUNCIONES MATEMÁTICAS", """
Puede escribir expresiones matemáticas directamente.

Ejemplos válidos:

sin(pi/2)

cos(0)

sqrt(16)

ln(e)

2*pi

2^3

3^2 + 1

Funciones disponibles:

sin(x)
cos(x)
tan(x)

sqrt(x)

ln(x)

log(x)

pi
e

También puede usar potencias con ^ o con **.

Ejemplo:

2^3  es igual a  2**3

Todas las funciones trigonométricas trabajan en radianes.
""", "#ffb347")

    tk.Button(ventana_exp, text="Cerrar", command=ventana_exp.destroy,
              bg=ACCENT, fg="white", relief="flat",
              font=("Segoe UI", 11, "bold"),
              padx=25, pady=10, cursor="hand2").pack(pady=15)


# -----------------------------------------------------------------------------
# 2.3  PANEL DE PROCEDIMIENTO PASO A PASO
# -----------------------------------------------------------------------------
# Recibe el diccionario 'datos' que devuelve spline_natural() y construye
# un widget de texto con scrollbar que muestra cada paso del cálculo:
# puntos ingresados → distancias h → momentos M → intervalo → resultado.
# -----------------------------------------------------------------------------

def mostrar_procedimiento(d):
    """Limpia frame_proc y rellena un Text widget con el procedimiento."""
    for w in frame_proc.winfo_children():
        w.destroy()

    texto    = tk.Text(frame_proc, bg=CARD, fg=TEXT,
                       font=("Consolas", 10), wrap="word",
                       relief="flat", padx=10, pady=10)
    scrollbar = tk.Scrollbar(frame_proc, command=texto.yview)
    texto.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    texto.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    contenido = """
==============================
PROCEDIMIENTO PASO A PASO
==============================

PUNTOS INGRESADOS

"""
    for i in range(len(d["x"])):
        contenido += f"P{i} = ({d['x'][i]}, {d['y'][i]})\n"

    contenido += "\nDISTANCIAS h\n\n"
    for i in range(len(d["h"])):
        contenido += (
            f"h{i} = {d['x'][i+1]} - {d['x'][i]}\n"
            f"h{i} = {d['h'][i]}\n\n"
        )

    contenido += "MOMENTOS Mi\n\n"
    for i in range(len(d["M"])):
        contenido += f"M{i} = {round(d['M'][i], 6)}\n"

    contenido += f"""
VALOR A INTERPOLAR

x = {d['valor']}

INTERVALO USADO

[{d['x1']}, {d['x2']}]

DATOS DEL INTERVALO

x1 = {d['x1']}
x2 = {d['x2']}
y1 = {d['y1']}
y2 = {d['y2']}
h  = {d['hi']}

M1 = {d['M1']}
M2 = {d['M2']}

FORMULA

S(x) = M1(x2-x)^3 / 6h
     + M2(x-x1)^3 / 6h
     + (y1-M1h^2/6)(x2-x)/h
     + (y2-M2h^2/6)(x-x1)/h

RESULTADO FINAL

S({d['valor']}) = {round(d['resultado'], 6)}
"""
    texto.insert("1.0", contenido)
    texto.config(state="disabled")


# -----------------------------------------------------------------------------
# 2.4  GRÁFICAS — SPLINE Y MOMENTOS Mᵢ
# -----------------------------------------------------------------------------
# Produce dos subplots dentro de la figura compartida 'fig':
#
#   Izquierda: curva del spline por tramos, nodos y punto interpolado.
#              Cada tramo tiene su propio color para distinguirlos.
#
#   Derecha:   gráfica de barras con los momentos Mᵢ.
#              Las barras de los extremos (M=0) se distinguen visualmente
#              de las interiores para reflejar la condición natural.
# -----------------------------------------------------------------------------

def graficar(x_pts, y_pts, M, h, valor, resultado):
    """Dibuja el spline y los momentos en la figura principal."""
    fig.clf()

    colores = ["#6c63ff", "#ff4d94", "#00e5a0", "#ffb347", "#00cfff"]

    # ── subplot izquierdo: spline ──────────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1)

    for i in range(len(x_pts) - 1):
        xs = np.linspace(x_pts[i], x_pts[i + 1], 200)
        ys = [evaluar_spline(x_pts, y_pts, M, h, t) for t in xs]
        c  = colores[i % len(colores)]
        ax1.plot(xs, ys, color=c, lw=2.5,
                 label=f"Tramo {i}  [{x_pts[i]}, {x_pts[i+1]}]")
        ax1.fill_between(xs, ys, alpha=0.08, color=c)

    ax1.scatter(x_pts, y_pts, s=90,
                color=CARD, edgecolors=ACCENT, linewidths=2,
                zorder=5, label="Nodos")

    for i in range(len(x_pts)):
        ax1.annotate(f"({x_pts[i]}, {y_pts[i]})",
                     xy=(x_pts[i], y_pts[i]),
                     xytext=(6, 6), textcoords="offset points",
                     fontsize=8, color=SUBTEXT)

    ax1.scatter([valor], [resultado],
                color=ACCENT2, s=200, marker="*", zorder=7,
                label=f"S({valor}) = {round(resultado, 4)}")
    ax1.axvline(valor,     color=ACCENT2, ls="--", alpha=0.3)
    ax1.axhline(resultado, color=ACCENT2, ls="--", alpha=0.3)

    ax1.set_title("Spline Cúbico Natural", fontsize=15, fontweight="bold")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True)
    ax1.legend(fontsize=8)

    # ── subplot derecho: momentos Mᵢ ──────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor(MI_BG)

    indices   = np.arange(len(M))
    colores_m = [ACCENT  if abs(m) < 1e-10 else ACCENT2 for m in M]
    borde_m   = [MI_ZERO_C if abs(m) < 1e-10 else MI_INT_C for m in M]

    bars = ax2.bar(indices, M, color=colores_m, edgecolor=borde_m,
                   linewidth=1.5, width=0.45, zorder=3)

    ax2.axhline(0, color=MI_ZERO, linewidth=1.8, zorder=4, linestyle="-")
    ax2.grid(True, which="major", color=MI_GRID, linewidth=1.0, alpha=1.0, zorder=0)
    ax2.minorticks_on()
    ax2.grid(True, which="minor", color=MI_GRID, linewidth=0.5, alpha=0.7, zorder=0)

    for spine in ax2.spines.values():
        spine.set_edgecolor(MI_SPINE)
        spine.set_linewidth(0.8)

    ax2.tick_params(colors=SUBTEXT, labelsize=8)
    ax2.xaxis.label.set_color(SUBTEXT)
    ax2.yaxis.label.set_color(SUBTEXT)

    mx    = max(M)
    mn    = min(M)
    rango = mx - mn if (mx - mn) != 0 else 1
    ax2.set_ylim(mn - rango * 0.28, mx + rango * 0.28)
    ax2.set_xlim(-0.7, len(M) - 0.3)
    ax2.set_xticks(indices)
    ax2.set_xticklabels(
        [f"M{i}\nx={x_pts[i]}" for i in indices],
        fontsize=8, color=SUBTEXT
    )

    offset = rango * 0.055
    for bar, val in zip(bars, M):
        cx    = bar.get_x() + bar.get_width() / 2
        ey    = val + offset if val >= 0 else val - offset
        va    = "bottom" if val >= 0 else "top"
        txt_c = MI_ZERO_C if abs(val) < 1e-10 else MI_INT_C
        ax2.text(cx, ey, f"{val:.4f}",
                 ha="center", va=va,
                 fontsize=9, color=txt_c, fontweight="bold")

    ax2.set_title("Momentos Mᵢ  (2das derivadas)",
                  fontsize=13, fontweight="bold", color=TEXT, pad=10)
    ax2.set_xlabel("Nodo")
    ax2.set_ylabel("Valor M")

    p1 = mpatches.Patch(color=ACCENT,  label="M = 0  (condición natural)")
    p2 = mpatches.Patch(color=ACCENT2, label="M ≠ 0  (nodo interior)")
    ax2.legend(handles=[p1, p2], fontsize=8,
               facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT)

    fig.tight_layout(pad=2.5)
    canvas.draw()


# -----------------------------------------------------------------------------
# 2.5  CALLBACKS DE LOS BOTONES
# -----------------------------------------------------------------------------
# calcular() — orquesta todo: lee entradas → llama a spline_natural()
#              → actualiza etiqueta de resultado → llama a mostrar_procedimiento()
#              → llama a graficar()
#
# limpiar()  — resetea todos los widgets a su estado inicial.
# -----------------------------------------------------------------------------

def calcular():
    """Lee los datos de la interfaz, ejecuta el método y actualiza la UI."""
    try:
        texto = entrada_puntos.get("1.0", tk.END).strip()
        valor = float(evaluar(entrada_valor.get()))

        x, y = [], []
        for fila in texto.split("\n"):
            datos = fila.split(",")
            if len(datos) != 2:
                messagebox.showerror("Error", "Use x,y por línea")
                return
            x.append(float(evaluar(datos[0])))
            y.append(float(evaluar(datos[1])))

        pares = sorted(zip(x, y))
        x = [p[0] for p in pares]
        y = [p[1] for p in pares]

        res, M, h, datos, err = spline_natural(x, y, valor)

        if err:
            messagebox.showerror("Error", err)
            return

        lbl_res.config(text=f"S({valor}) = {round(res, 6)}", fg=SUCCESS)
        mostrar_procedimiento(datos)
        graficar(x, y, M, h, valor, res)

    except Exception as e:
        messagebox.showerror("Error", str(e))


def limpiar():
    """Vacía todos los campos y borra gráficas y procedimiento."""
    entrada_puntos.delete("1.0", tk.END)
    entrada_valor.delete(0, tk.END)
    lbl_res.config(text="Ingrese datos y calcule", fg=SUBTEXT)
    for w in frame_proc.winfo_children():
        w.destroy()
    fig.clf()
    canvas.draw()


# -----------------------------------------------------------------------------
# 2.6  CONSTRUCCIÓN DE LA VENTANA PRINCIPAL
# -----------------------------------------------------------------------------
# Layout: dos columnas
#   • Panel izquierdo (PANEL, ancho fijo 380 px):
#       título, inputs, botones, etiqueta de resultado, procedimiento.
#   • Panel derecho (BG, expansible):
#       figura de matplotlib embebida con FigureCanvasTkAgg.
# -----------------------------------------------------------------------------

ventana = tk.Tk()
ventana.title("Spline Cubico Natural")
ventana.state("zoomed")
ventana.configure(bg=BG)

# — columna izquierda —
panel = tk.Frame(ventana, bg=PANEL, width=380)
panel.pack(side=tk.LEFT, fill=tk.Y)
panel.pack_propagate(False)

tk.Label(panel, text="Spline Cubico Natural",
         bg=PANEL, fg=ACCENT,
         font=("Segoe UI", 20, "bold")).pack(pady=(20, 10))

tk.Label(panel, text="Puntos (x,y)",
         bg=PANEL, fg=SUBTEXT,
         font=("Segoe UI", 10)).pack(anchor="w", padx=15)

entrada_puntos = tk.Text(panel, height=7, bg=CARD, fg=TEXT,
                         insertbackground=TEXT, relief="flat",
                         font=("Consolas", 11))
entrada_puntos.pack(fill=tk.X, padx=15, pady=5)
entrada_puntos.insert(tk.END, "1,2\n2,3\n3,5\n4,4\n5,6")

tk.Label(panel, text="Valor a interpolar",
         bg=PANEL, fg=SUBTEXT,
         font=("Segoe UI", 10)).pack(anchor="w", padx=15)

entrada_valor = tk.Entry(panel, bg=CARD, fg=TEXT,
                         insertbackground=TEXT, relief="flat",
                         font=("Consolas", 12))
entrada_valor.pack(fill=tk.X, padx=15, pady=5)
entrada_valor.insert(0, "2.5")

def crear_boton(texto, comando, color):
    return tk.Button(panel, text=texto, command=comando,
                     bg=color, fg="white", relief="flat",
                     font=("Segoe UI", 11, "bold"),
                     pady=10, cursor="hand2")

crear_boton("Calcular",                      calcular,           ACCENT).pack(fill=tk.X, padx=15, pady=5)
crear_boton("Limpiar",                       limpiar,            CARD2).pack(fill=tk.X, padx=15, pady=5)
crear_boton("Explicación del Procedimiento", mostrar_explicacion, "#5a4fcf").pack(fill=tk.X, padx=15, pady=5)

lbl_res = tk.Label(panel, text="Ingrese datos y calcule",
                   bg=CARD, fg=SUBTEXT,
                   font=("Segoe UI", 11, "bold"), pady=12)
lbl_res.pack(fill=tk.X, padx=15, pady=10)

tk.Label(panel, text="Procedimiento paso a paso",
         bg=PANEL, fg=TEXT,
         font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=15)

frame_proc = tk.Frame(panel, bg=PANEL)
frame_proc.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

# — columna derecha —
frame_der = tk.Frame(ventana, bg=BG)
frame_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

fig    = plt.Figure(figsize=(11, 5), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=frame_der)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

ventana.mainloop()