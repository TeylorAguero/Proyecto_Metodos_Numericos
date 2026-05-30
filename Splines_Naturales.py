# =============================================================================
# SPLINE CÚBICO NATURAL — Interfaz Profesional
# =============================================================================

import subprocess
import sys
import math

# =============================================================================
# SECCIÓN 1 — LÓGICA MATEMÁTICA
# =============================================================================

permitidas = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "sqrt": math.sqrt, "ln": math.log, "log": math.log10,
    "pi": math.pi, "e": math.e, "abs": abs
}

def evaluar(expr):
    expr = expr.strip().lower().replace("^", "**")
    return eval(expr, {"__builtins__": {}}, permitidas)


def resolver_sistema_gauss(A, b):
    n = len(b)
    for i in range(n):
        mayor = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[mayor][i]):
                mayor = k
        A[i], A[mayor] = A[mayor], A[i]
        b[i], b[mayor] = b[mayor], b[i]
        if A[i][i] == 0:
            return None
        for k in range(i + 1, n):
            factor = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
            b[k] -= factor * b[i]
    x = [0] * n
    for i in range(n - 1, -1, -1):
        suma = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - suma) / A[i][i]
    return x


def spline_natural(x, y, valor):
    n = len(x)
    if n < 3:
        return None, None, None, None, "Se necesitan al menos 3 puntos."
    h = []
    for i in range(n - 1):
        hi = x[i + 1] - x[i]
        if hi == 0:
            return None, None, None, None, "No puede haber x repetidos."
        h.append(hi)

    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    A[0][0] = 1
    A[n - 1][n - 1] = 1
    for i in range(1, n - 1):
        A[i][i - 1] = h[i - 1]
        A[i][i]     = 2 * (h[i - 1] + h[i])
        A[i][i + 1] = h[i]
        b[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    M = resolver_sistema_gauss(A, b)
    if M is None:
        return None, None, None, None, "No se pudo resolver."

    pos = -1
    for i in range(n - 1):
        if x[i] <= valor <= x[i + 1]:
            pos = i
            break
    if pos == -1:
        return None, None, None, None, "Valor fuera del rango."

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

for pkg in ["matplotlib", "numpy"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import rcParams
import numpy as np

# ── Paleta "dusk" — tonos medios cálidos, sin blanco ni negro puro ────────────
BG_WIN   = "#2B2D3E"   # ventana: gris-azul oscuro cálido
PANEL_BG = "#242636"   # panel izquierdo
CARD_BG  = "#1E2030"   # campos de entrada
CARD2    = "#2E3046"   # botón limpiar / tarjetas secundarias
BORDER   = "#3E4060"   # bordes
TEXT_H   = "#E8EAF6"   # títulos lavanda claro
TEXT_B   = "#C5C8E0"   # cuerpo
TEXT_S   = "#7E84A8"   # subtexto apagado
ACCENT   = "#5B7BF5"   # azul índigo
ACCENT_H = "#4A68E0"   # hover
ACCENT2  = "#F07090"   # rosa salmón — punto interpolado
ACCENT3  = "#4FC3D4"   # cyan decorativo
SUCCESS  = "#3DCFB8"   # verde-agua — resultado
DANGER   = "#E05555"

PLOT_COLORS = [
    "#5B7BF5",  # azul índigo
    "#F07090",  # rosa salmón
    "#3DCFB8",  # turquesa
    "#F0A050",  # naranja cálido
    "#A07BE0",  # violeta
    "#50C8A0",  # verde menta
    "#E07070",  # coral
    "#50A8D8",  # azul acero
]

# ── Matplotlib: tema dusk ─────────────────────────────────────────────────────
rcParams.update({
    "figure.facecolor":  "#2B2D3E",
    "axes.facecolor":    "#232535",
    "axes.edgecolor":    "#3E4060",
    "axes.labelcolor":   "#7E84A8",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       "#7E84A8",
    "ytick.color":       "#7E84A8",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "text.color":        "#C5C8E0",
    "grid.color":        "#343650",
    "grid.alpha":        1.0,
    "grid.linewidth":    0.6,
    "legend.facecolor":  "#242636",
    "legend.edgecolor":  "#3E4060",
    "legend.labelcolor": "#C5C8E0",
    "legend.fontsize":   8.5,
    "font.family":       "DejaVu Sans",
    "font.size":         9.5,
})


# ── Ventana de teoría ─────────────────────────────────────────────────────────
def mostrar_explicacion():
    win = tk.Toplevel(ventana)
    win.title("Teoría — Spline Cúbico Natural")
    win.geometry("820x640")
    win.configure(bg=PANEL_BG)

    tk.Label(win, text="Fundamentos del Método",
             bg=PANEL_BG, fg=ACCENT, font=("Segoe UI", 18, "bold")).pack(pady=(22, 2))
    tk.Label(win, text="Spline Cúbico Natural",
             bg=PANEL_BG, fg=TEXT_S, font=("Segoe UI", 10)).pack(pady=(0, 14))

    outer = tk.Frame(win, bg=PANEL_BG)
    outer.pack(fill=tk.BOTH, expand=True, padx=24, pady=4)

    cv  = tk.Canvas(outer, bg=PANEL_BG, highlightthickness=0)
    vsb = tk.Scrollbar(outer, orient="vertical", command=cv.yview)
    cv.configure(yscrollcommand=vsb.set)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)
    cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    inner = tk.Frame(cv, bg=PANEL_BG)
    wid   = cv.create_window((0, 0), window=inner, anchor="nw")
    cv.bind("<Configure>", lambda e: cv.itemconfig(wid, width=e.width))
    inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
    cv.bind_all("<MouseWheel>", lambda e: cv.yview_scroll(int(-1*(e.delta/120)), "units"))

    def bloque(numero, titulo, cuerpo, color):
        f = tk.Frame(inner, bg=CARD2, highlightbackground=BORDER, highlightthickness=1)
        f.pack(fill=tk.X, padx=4, pady=6)
        head = tk.Frame(f, bg=CARD2)
        head.pack(fill=tk.X, padx=18, pady=(14, 6))
        tk.Label(head, text=numero, bg=CARD2, fg=color,
                 font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(head, text=titulo, bg=CARD2, fg=TEXT_H,
                 font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT)
        tk.Frame(f, bg=BORDER, height=1).pack(fill=tk.X, padx=18)
        tk.Label(f, text=cuerpo, bg=CARD2, fg=TEXT_B,
                 font=("Consolas", 10), justify="left", anchor="w",
                 wraplength=700, padx=18, pady=12).pack(anchor="w", fill=tk.X)

    bloque("01", "Introducción", """\
El spline cúbico natural interpola n puntos con n−1 polinomios de grado 3.
La curva es suave: primera y segunda derivada son continuas en cada nodo
interior. Esto evita las oscilaciones del polinomio de Lagrange de alto grado.""", ACCENT)

    bloque("02", "Distancias  hᵢ", """\
Para cada par de nodos consecutivos:

    hᵢ = x(i+1) − xᵢ

Determinan el ancho de cada tramo y aparecen como coeficientes
en el sistema tridiagonal.""", "#7B2FBE")

    bloque("03", "Fórmula de evaluación", """\
En el intervalo [x₁, x₂] con momentos M₁, M₂ y paso h:

    S(x) =  M₁·(x₂−x)³ / 6h
          + M₂·(x−x₁)³ / 6h
          + (y₁ − M₁h²/6)·(x₂−x) / h
          + (y₂ − M₂h²/6)·(x−x₁) / h""", SUCCESS)

    bloque("04", "Cómo escribir expresiones matemáticas", """\
Puede escribir expresiones directamente en los campos.
Aquí se muestra cómo representar cada operación:

  ┌─────────────────────────────────────────────────────────┐
  │  LO QUE QUIERE          →   LO QUE ESCRIBE             │
  ├─────────────────────────────────────────────────────────┤
  │  √16  (raíz cuadrada)   →   sqrt(16)                   │
  │  2³   (potencia)        →   2^3   ó   2**3             │
  │  x²   (elevar al cuadrado) →  x^2  ó  x**2            │
  │  π    (pi)              →   pi                         │
  │  e    (número de Euler) →   e                          │
  │  |x|  (valor absoluto)  →   abs(x)                     │
  │  sen  (seno)            →   sin(x)   [en radianes]     │
  │  cos  (coseno)          →   cos(x)   [en radianes]     │
  │  tan  (tangente)        →   tan(x)   [en radianes]     │
  │  ln   (log natural)     →   ln(x)                      │
  │  log  (log base 10)     →   log(x)                     │
  └─────────────────────────────────────────────────────────┘

  Ejemplos completos:
    √(2·π)        →   sqrt(2*pi)
    sen(π/2)      →   sin(pi/2)
    e²            →   e^2
    3² + √5       →   3^2 + sqrt(5)""", "#FF9F1C")

    tk.Button(win, text="Cerrar", command=win.destroy,
              bg=ACCENT, fg="white", relief="flat",
              font=("Segoe UI", 10, "bold"),
              padx=28, pady=9, cursor="hand2").pack(pady=20)


# ── Gráfica ───────────────────────────────────────────────────────────────────
def graficar(x_pts, y_pts, M, h, valor, resultado):
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)

    # Tramos
    for i in range(len(x_pts) - 1):
        xs = np.linspace(x_pts[i], x_pts[i + 1], 300)
        ys = [evaluar_spline(x_pts, y_pts, M, h, t) for t in xs]
        c  = PLOT_COLORS[i % len(PLOT_COLORS)]
        ax.plot(xs, ys, color=c, lw=2.5,
                label=f"Tramo {i}  [ {x_pts[i]}, {x_pts[i+1]} ]",
                solid_capstyle="round", zorder=3)
        ax.fill_between(xs, ys, alpha=0.07, color=c, zorder=2)

    # Nodos conocidos
    ax.scatter(x_pts, y_pts,
               s=90, zorder=5,
               color="#1E2030", edgecolors=ACCENT, linewidths=2.0,
               label="Nodos conocidos")

    # Etiquetas de nodos
    for xi, yi in zip(x_pts, y_pts):
        ax.annotate(f"  ({xi}, {yi})",
                    xy=(xi, yi), xytext=(5, 6),
                    textcoords="offset points",
                    fontsize=8, color=TEXT_S)

    # Punto interpolado
    ax.scatter([valor], [resultado],
               s=180, zorder=6, marker="*",
               color=ACCENT2,
               label=f"S({valor}) = {round(resultado, 6)}")

    # Líneas de referencia
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.plot([valor, valor], [ylim[0], resultado],
            color=ACCENT2, ls="--", lw=1.0, alpha=0.5, zorder=1)
    ax.plot([xlim[0], valor], [resultado, resultado],
            color=ACCENT2, ls="--", lw=1.0, alpha=0.5, zorder=1)

    # Anotación del resultado
    ax.annotate(
        f"  S({valor}) = {round(resultado, 6)}",
        xy=(valor, resultado),
        xytext=(12, 14), textcoords="offset points",
        fontsize=9, color=ACCENT2, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#1E2030",
                  ec=ACCENT2, lw=1.0, alpha=0.92)
    )

    # Formato de ejes
    ax.set_title("Spline Cúbico Natural",
                 fontsize=14, fontweight="bold",
                 color=TEXT_H, pad=14)
    ax.set_xlabel("x", fontsize=10, color=TEXT_S, labelpad=8)
    ax.set_ylabel("S(x)", fontsize=10, color=TEXT_S, labelpad=8)
    ax.grid(True, which="major", linewidth=0.7)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.5)
    ax.minorticks_on()
    ax.tick_params(which="both", length=0)

    # Leyenda
    leg = ax.legend(fontsize=8.5, framealpha=0.95,
                    loc="best", borderpad=0.8,
                    handlelength=1.8, labelspacing=0.45)
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_edgecolor(BORDER)

    fig.tight_layout(pad=2.5)
    canvas.draw()


# ── Callbacks ─────────────────────────────────────────────────────────────────
def calcular():
    try:
        texto = entrada_puntos.get("1.0", tk.END).strip()
        valor = float(evaluar(entrada_valor.get()))

        x, y = [], []
        for fila in texto.split("\n"):
            if not fila.strip():
                continue
            cols = fila.split(",")
            if len(cols) != 2:
                messagebox.showerror("Formato incorrecto",
                                     "Ingrese un par  x,y  por línea.\nEjemplo:  1,2")
                return
            x.append(float(evaluar(cols[0])))
            y.append(float(evaluar(cols[1])))

        pares = sorted(zip(x, y))
        x = [p[0] for p in pares]
        y = [p[1] for p in pares]

        res, M, h, datos, err = spline_natural(x, y, valor)
        if err:
            messagebox.showerror("Error", err)
            return

        lbl_eq.config(text=f"S( {valor} )", fg=TEXT_S)
        lbl_val.config(text=f"{round(res, 8)}", fg=SUCCESS)
        lbl_rango.config(
            text=f"Intervalo  [{datos['x1']},  {datos['x2']}]",
            fg=TEXT_S)

        graficar(x, y, M, h, valor, res)

    except Exception as e:
        messagebox.showerror("Error", str(e))


def limpiar():
    entrada_puntos.delete("1.0", tk.END)
    entrada_puntos.insert(tk.END, "1,2\n2,3\n3,5\n4,4\n5,6")
    entrada_valor.delete(0, tk.END)
    entrada_valor.insert(0, "2.5")
    lbl_eq.config(text="S( x )", fg=BORDER)
    lbl_val.config(text="—", fg=TEXT_S)
    lbl_rango.config(text="", fg=TEXT_S)
    fig.clf()
    canvas.draw()


# ── Helpers de widgets ────────────────────────────────────────────────────────
def sep(parent, pad_y=10):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, padx=20, pady=pad_y)

def label_sec(parent, txt):
    tk.Label(parent, text=txt, bg=PANEL_BG, fg=TEXT_S,
             font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=20, pady=(10, 3))

def mk_entry(parent):
    return tk.Entry(parent, bg=CARD_BG, fg=TEXT_H,
                    insertbackground=ACCENT, relief="flat",
                    font=("Consolas", 12),
                    highlightbackground=BORDER,
                    highlightcolor=ACCENT,
                    highlightthickness=1)

def mk_text(parent, h=8):
    return tk.Text(parent, height=h,
                   bg=CARD_BG, fg=TEXT_H,
                   insertbackground=ACCENT, relief="flat",
                   font=("Consolas", 11),
                   highlightbackground=BORDER,
                   highlightcolor=ACCENT,
                   highlightthickness=1,
                   selectbackground=ACCENT,
                   selectforeground="white")

def mk_btn(parent, txt, cmd, bg=ACCENT, fg="white"):
    def on_enter(e): b.config(bg=ACCENT_H if bg == ACCENT else CARD2)
    def on_leave(e): b.config(bg=bg)
    b = tk.Button(parent, text=txt, command=cmd,
                  bg=bg, fg=fg, relief="flat",
                  font=("Segoe UI", 10, "bold"),
                  pady=10, cursor="hand2",
                  activebackground=ACCENT_H,
                  activeforeground="white",
                  bd=0)
    b.pack(fill=tk.X, padx=20, pady=3)
    b.bind("<Enter>", on_enter)
    b.bind("<Leave>", on_leave)
    return b


# ── Ventana principal ─────────────────────────────────────────────────────────
ventana = tk.Tk()
ventana.title("Spline Cúbico Natural")
ventana.state("zoomed")
ventana.configure(bg=BG_WIN)

# ══ PANEL IZQUIERDO ══════════════════════════════════════════════════════════
panel = tk.Frame(ventana, bg=PANEL_BG, width=310)
panel.pack(side=tk.LEFT, fill=tk.Y)
panel.pack_propagate(False)

# Borde derecho del panel
tk.Frame(ventana, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y)

# Barra superior de acento
tk.Frame(panel, bg=ACCENT, height=4).pack(fill=tk.X)

# Logo / Título
header = tk.Frame(panel, bg=PANEL_BG)
header.pack(fill=tk.X, padx=20, pady=(20, 6))

badge = tk.Frame(header, bg=ACCENT, padx=8, pady=4)
badge.pack(anchor="w")
tk.Label(badge, text="MÉTODOS NUMÉRICOS", bg=ACCENT, fg="white",
         font=("Segoe UI", 7, "bold")).pack()

tk.Label(panel, text="Spline Cúbico",
         bg=PANEL_BG, fg=TEXT_H,
         font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=20, pady=(8, 0))
tk.Label(panel, text="Natural",
         bg=PANEL_BG, fg=ACCENT,
         font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=20, pady=(0, 4))
tk.Label(panel, text="Interpolación por tramos cúbicos",
         bg=PANEL_BG, fg=TEXT_S,
         font=("Segoe UI", 8)).pack(anchor="w", padx=20)

sep(panel, 14)

# Entradas
label_sec(panel, "PUNTOS  ( x , y )")
tk.Label(panel, text="Un par por línea  —  ejemplo:  1,2",
         bg=PANEL_BG, fg=TEXT_S, font=("Segoe UI", 7)).pack(anchor="w", padx=20, pady=(0, 4))

entrada_puntos = mk_text(panel, h=8)
entrada_puntos.pack(fill=tk.X, padx=20, pady=(0, 6))
entrada_puntos.insert(tk.END, "1,2\n2,3\n3,5\n4,4\n5,6")

label_sec(panel, "VALOR A INTERPOLAR")
entrada_valor = mk_entry(panel)
entrada_valor.pack(fill=tk.X, padx=20, pady=(0, 12))
entrada_valor.insert(0, "2.5")

# Botones
mk_btn(panel, "▶   Calcular",           calcular, ACCENT)
mk_btn(panel, "↺   Limpiar",            limpiar,  CARD2, TEXT_H)
mk_btn(panel, "?   Ver teoría",         mostrar_explicacion, "#4A3F8A")

sep(panel, 14)

# Tarjeta de resultado
res_wrap = tk.Frame(panel, bg=CARD_BG,
                    highlightbackground=BORDER, highlightthickness=1)
res_wrap.pack(fill=tk.X, padx=20)
tk.Frame(res_wrap, bg=SUCCESS, height=3).pack(fill=tk.X)

res_inner = tk.Frame(res_wrap, bg=CARD_BG)
res_inner.pack(fill=tk.X, padx=16, pady=14)

tk.Label(res_inner, text="RESULTADO", bg=CARD_BG, fg=TEXT_S,
         font=("Segoe UI", 7, "bold")).pack(anchor="w")

lbl_eq = tk.Label(res_inner, text="S( x )", bg=CARD_BG, fg=BORDER,
                   font=("Segoe UI", 10))
lbl_eq.pack(anchor="w", pady=(2, 0))

lbl_val = tk.Label(res_inner, text="—", bg=CARD_BG, fg=TEXT_S,
                    font=("Consolas", 20, "bold"))
lbl_val.pack(anchor="w", pady=(2, 4))

lbl_rango = tk.Label(res_inner, text="", bg=CARD_BG, fg=TEXT_S,
                      font=("Segoe UI", 8))
lbl_rango.pack(anchor="w")

# Pie del panel
tk.Label(panel, text="Eliminación de Gauss  ·  Spline natural",
         bg=PANEL_BG, fg=BORDER,
         font=("Segoe UI", 7)).pack(side=tk.BOTTOM, pady=10)


# ══ ÁREA DE GRÁFICA ══════════════════════════════════════════════════════════
frame_der = tk.Frame(ventana, bg=BG_WIN)
frame_der.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Barra superior del área de gráfica
topbar = tk.Frame(frame_der, bg=BG_WIN, height=38)
topbar.pack(fill=tk.X)
tk.Label(topbar, text="Gráfica del spline",
         bg=BG_WIN, fg=TEXT_S,
         font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT, padx=20, pady=12)

fig    = plt.Figure(figsize=(10, 6), dpi=100, facecolor="#2B2D3E")
canvas = FigureCanvasTkAgg(fig, master=frame_der)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 18))

ventana.mainloop()