import base64
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import analisis as an

st.set_page_config(page_title="TDAH · Proyecto final", layout="wide")
OSC, TEAL, AMB, ROJ = "#01282E", "#0CC0DF", "#E9C46A", "#E76F51"
MENU = ["#0CC0DF", "#0097B2", "#027387", "#005463", "#02414C", "#01282E"]
COL_SEG = {"Enfocados constantes": "#0097B2", "Esforzados con atención corta": AMB, "Dispersos con poco sueño": ROJ}
PAGINAS = [("situacion", "1. Situación y datos"), ("descriptiva", "2. Estadística descriptiva"), ("limpieza", "3. Limpieza"),
           ("inferencial", "4. Inferencial"), ("kmeans", "5. K-Means"), ("dashboard", "6. Dashboard")]
sns.set_theme(style="whitegrid")


@st.cache_data
def cargar():
    crudo = an.generar_datos()
    return crudo, an.limpiar(crudo)


def b64(nombre):
    return base64.b64encode((Path(__file__).parent / nombre).read_bytes()).decode()


def ir(k):
    st.session_state.pag = k


def tarjeta(titulo, valor, color=MENU[1]):
    st.markdown(f"""<div style="background:{color};padding:14px 16px;border-radius:8px;color:white;text-align:center">
        <div style="font-size:.9rem;opacity:.92">{titulo}</div><div style="font-size:1.5rem;font-weight:700">{valor}</div></div>""",
                unsafe_allow_html=True)


def mostrar(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def fmt_p(p):
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def elegir(clave, opciones=an.VARS, etiqueta="Variable"):
    return st.selectbox(etiqueta, opciones, format_func=lambda c: an.ETIQ[c], key=clave)


def encabezado(msg, grande=False):
    """Título TDAH y el perro con su globo de texto, como en el diseño."""
    st.markdown("""<div class="tit"><h1>T D A H</h1><p>Trastorno por Déficit de Atención e Hiperactividad</p></div>""", unsafe_allow_html=True)
    alto = 190 if grande else 120
    st.markdown(f"""<div class="fila"><img src="data:image/png;base64,{PERRO}" style="height:{alto}px">
        <div class="burbuja">{msg}</div></div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- gráficas reutilizables
def g_hist(d, var, ax):
    sns.histplot(d[var].dropna(), kde=True, color=MENU[1], ax=ax)
    ax.set_title("Histograma"); ax.set_xlabel(an.ETIQ[var])


def g_caja(d, var, ax):
    sns.boxplot(data=d, x="taller_gestion_tiempo", y=var, hue="taller_gestion_tiempo", order=["No", "Sí"],
                palette={"No": AMB, "Sí": MENU[1]}, legend=False, ax=ax)
    ax.set_title("Con taller y sin taller"); ax.set_xlabel("Participó en el taller de gestión del tiempo"); ax.set_ylabel(an.ETIQ[var])


def g_reg(d, x, ax):
    sns.regplot(data=d, x=x, y="calificacion", scatter_kws={"alpha": .55, "color": MENU[1]}, line_kws={"color": ROJ}, ax=ax)
    ax.set_title("Regresión lineal simple"); ax.set_xlabel(an.ETIQ[x]); ax.set_ylabel(an.ETIQ["calificacion"])


def g_seg(seg, cent, colores, ax):
    sns.scatterplot(data=seg, x="minutos_atencion", y="horas_estudio_semana", hue="segmento", palette=colores, s=55, ax=ax)
    for _, f in cent.iterrows():
        ax.scatter(f["minutos_atencion"], f["horas_estudio_semana"], marker="*", s=380, color=colores[f["segmento"]], edgecolor="black", zorder=10)
    ax.set_title("Segmentos y centroides"); ax.set_xlabel(an.ETIQ["minutos_atencion"]); ax.set_ylabel(an.ETIQ["horas_estudio_semana"])
    ax.legend(title="", fontsize=8)


# ---------------------------------------------------------------- estilo y menú lateral
PERRO = b64("perro.png")
pag = st.session_state.setdefault("pag", "inicio")
activo = [k for k, _ in PAGINAS].index(pag) + 1 if pag != "inicio" else 0
css_menu = "".join(f".st-key-m_{i + 1} button{{background:{c} !important}}" for i, c in enumerate(MENU))
css_act = f".st-key-m_{activo} button{{box-shadow:inset 9px 0 0 #fff !important}}" if activo else ""
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&display=swap');
header[data-testid="stHeader"]{{background:transparent}}
.block-container{{max-width:1100px;padding-top:1.5rem}}
section[data-testid="stSidebar"]{{background:{OSC};min-width:300px;max-width:300px}}
[data-testid="stSidebarHeader"]{{display:none}}
[data-testid="stSidebarUserContent"]{{padding:0}}
.st-key-menu [data-testid="stVerticalBlock"]{{gap:0}}
.st-key-menu [data-testid="stButton"]{{width:100%}}
.st-key-menu button{{width:100%;height:78px;border:0 !important;border-radius:0 !important;box-shadow:none}}
.st-key-menu button p{{color:#fff !important;font-family:'Libre Baskerville',Georgia,serif;font-size:1rem}}
.st-key-menu button:hover{{filter:brightness(1.12)}}
{css_menu}{css_act}
.st-key-filtro{{padding:18px 22px}}
.st-key-filtro label p, .st-key-filtro p, .st-key-filtro div{{color:#fff}}
.tit{{border-top:2px solid {OSC};border-bottom:2px solid {OSC};text-align:center;padding:4px 0 8px 0;margin-bottom:6px}}
.tit h1{{font-family:'Libre Baskerville',Georgia,serif;letter-spacing:.45em;font-size:2.6rem;margin:0;padding:0;color:{OSC}}}
.tit p{{font-family:'Libre Baskerville',Georgia,serif;margin:2px 0 0 0;font-size:.95rem;color:{OSC}}}
.fila{{display:flex;align-items:center;gap:20px;margin:16px 0 18px 0}}
.burbuja{{position:relative;background:{OSC};color:#fff;padding:14px 18px;border-radius:8px;max-width:620px;
  font-family:'Libre Baskerville',Georgia,serif;font-size:.95rem;line-height:1.5}}
.burbuja::before{{content:"";position:absolute;left:-10px;top:50%;margin-top:-9px;border:9px solid transparent;border-left:0;border-right-color:{OSC}}}
.pie{{display:flex;justify-content:space-around;flex-wrap:wrap;gap:8px;border-top:1.5px solid {OSC};margin-top:36px;padding-top:12px;
  font-family:'Libre Baskerville',Georgia,serif;font-size:.9rem;color:{OSC}}}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    with st.container(key="menu"):
        for i, (k, txt) in enumerate(PAGINAS, 1):
            st.button(txt, key=f"m_{i}", on_click=ir, args=(k,))
    with st.container(key="filtro"):
        sem = st.slider("Filtro de semestre", 2, 9, (2, 9))
        st.caption("Se aplica a todas las secciones, excepto Limpieza.")

crudo, limpio = cargar()
crudo_f, limpio_f = crudo[crudo.semestre.between(*sem)], limpio[limpio.semestre.between(*sem)]
if len(limpio_f) < 30:
    st.warning("Hay muy pocos estudiantes con ese filtro. Amplía el rango de semestres.")
    st.stop()
if pag != "inicio":
    st.button("Inicio", key="b_inicio", on_click=ir, args=("inicio",))


# ---------------------------------------------------------------- páginas
def p_inicio():
    encabezado("El TDAH no consiste en la falta total de atención, sino en una dificultad para regularla, "
               "lo que puede provocar el hiperfoco.", grande=True)
    st.write("Proyecto final de Análisis y Exploración de Datos (INC-2505). Elige una sección en el menú de la izquierda para recorrer el análisis.")


def p_situacion():
    encabezado("Este proyecto no diagnostica a nadie: usa datos simulados para entender perfiles de atención y decidir qué apoyo ofrecer.")
    st.subheader("Situación")
    st.write("Un centro de orientación psicopedagógica universitario recibe cada semestre a estudiantes que dicen perder la concentración con facilidad. "
             "Quiere ofrecer apoyo (talleres y tutorías), pero sus **recursos son limitados** y no puede dar lo mismo a todos. "
             "Cuenta con registros de **300 estudiantes** sobre sus hábitos de estudio, sueño, uso del celular y atención sostenida, "
             "además de su calificación promedio. El TDAH es un trastorno del neurodesarrollo que **solo puede diagnosticar un profesional**: "
             "aquí se estudian perfiles de regulación de la atención, no diagnósticos.")
    st.info("**Pregunta de decisión:** ¿qué apoyo conviene ofrecer a cada perfil de regulación de la atención para mejorar su desempeño?")
    st.subheader("Descripción del dataset")
    st.dataframe(pd.DataFrame({
        "Variable": ["id_estudiante", "semestre", "taller_gestion_tiempo", "horas_estudio_semana", "horas_sueno",
                     "minutos_atencion", "uso_celular_horas", "calificacion"],
        "Tipo": ["Texto (código anónimo)", "Numérica discreta", "Categórica (Sí / No)", "Numérica continua", "Numérica continua",
                 "Numérica continua", "Numérica continua", "Numérica continua"],
        "Descripción": ["Identificador sin nombre", "Semestre que cursa (2 a 9)", "Participó en el taller de gestión del tiempo",
                        "Horas de estudio fuera de clase", "Horas de sueño por noche", "Minutos que se concentra antes de distraerse",
                        "Horas diarias de uso del celular", "Calificación promedio, escala 0 a 100"]}), hide_index=True)
    st.write("**Fuente:** datos **simulados por la alumna** con semilla fija (42), diseñados con problemas de calidad a propósito "
             "(valores faltantes y capturas erróneas) para practicar la limpieza.")
    st.subheader("Consideración ética y protocolo de manejo de datos")
    st.markdown("""
- **Datos simulados:** no corresponden a personas reales, por lo que no se trata ningún dato personal.
- **Si fueran reales:** cada estudiante tendría un código anónimo (nunca su nombre), se pediría consentimiento informado y se daría un aviso de privacidad.
- **Datos sensibles:** no se recolectan ni se infieren diagnósticos de salud (por ejemplo, de TDAH). Los segmentos describen hábitos, no condiciones.
- **Mínimo necesario y reporte agregado:** solo se usan las variables indispensables y se reportan grupos, no personas.
- **Uso de los resultados:** solo para ofrecer apoyo voluntario, con revisión humana; nunca para sancionar ni limitar oportunidades.
- **Acceso:** los datos reales tendrían acceso restringido y no se publicarían en repositorios ni aplicaciones abiertas.
""")
    st.write("**Por qué sirve este dataset:** incluye hábitos medibles que podrían influir en la calificación, un grupo con taller y otro sin taller "
             "(para comparar) y suficientes estudiantes para encontrar perfiles distintos.")
    st.write("**Primeros registros (sin limpiar):**")
    st.dataframe(crudo_f.head(10), hide_index=True)


def p_descriptiva():
    encabezado("Antes de modelar hay que mirar los datos: sus centros, su dispersión y los valores que no encajan.")
    st.subheader("Estadística descriptiva y exploratoria (datos originales)")
    st.dataframe(an.resumen(crudo_f))
    var = elegir("v_expl", etiqueta="Variable a explorar")
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        g_hist(crudo_f, var, ax)
        mostrar(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        sns.boxplot(x=crudo_f[var], color=AMB, ax=ax)
        ax.set_title("Diagrama de caja"); ax.set_xlabel(an.ETIQ[var])
        mostrar(fig)
    q1, q3, iqr, li, ls = an.limites_tukey(crudo_f[var])
    n_at = int(((crudo_f[var] < li) | (crudo_f[var] > ls)).sum())
    st.write(f"**Regla de Tukey:** IQR = {iqr:.2f}; valores fuera de [{li:.2f}, {ls:.2f}] son atípicos. En esta variable hay **{n_at}**.")
    st.dataframe(an.atipicos(crudo_f))
    st.markdown("""**Anomalías propias del problema:**
- *Uso de celular:* valores de 15 a 22 horas al día son imposibles para alguien que duerme y estudia; son errores de captura.
- *Horas de estudio:* valores de 55 a 80 por semana sugieren que se registraron las horas del mes y no las de la semana.
- *Calificación:* valores mayores a 100 quedan fuera de la escala.
- *Sueño y atención:* no tienen atípicos, pero sí **valores faltantes**.""")
    st.subheader("Comparación entre subgrupos: con taller y sin taller")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    g_caja(crudo_f, var, ax)
    mostrar(fig)
    med = crudo_f.groupby("taller_gestion_tiempo")[var].median()
    st.write(f"La mediana de **{an.ETIQ[var].lower()}** es {med['Sí']:.2f} con taller y {med['No']:.2f} sin taller "
             f"(diferencia de {med['Sí'] - med['No']:+.2f}). Esto orienta qué comparar formalmente en la sección Inferencial.")


def p_limpieza():
    encabezado("Los datos reales casi nunca llegan limpios: limpiarlos bien cambia las conclusiones.")
    st.subheader("Preparación y limpieza")
    st.dataframe(pd.DataFrame({"Faltantes antes": crudo[an.VARS].isna().sum().values, "Faltantes después": limpio[an.VARS].isna().sum().values,
                               "Atípicos antes": an.atipicos(crudo)["atípicos"].values, "Atípicos después": an.atipicos(limpio)["atípicos"].values},
                              index=[an.ETIQ[c] for c in an.VARS]))
    st.markdown("""**Técnicas aplicadas y por qué:**
- **Imputación por mediana** (faltantes de sueño y atención): la mediana no se distorsiona con valores extremos, a diferencia de la media, y no se eliminan filas para no perder estudiantes.
- **Capping con los límites de Tukey** (atípicos): conserva cada registro y limita el daño de los errores de captura; eliminarlos habría descartado estudiantes por un solo dato mal capturado. La calificación además se limita a 100.
- **Estandarización z-score** (para K-means): el algoritmo usa distancias, así que las variables deben tener la misma escala. Se eligió z-score y no min-max porque min-max depende de los extremos.""")
    var = elegir("v_limp", etiqueta="Variable para comparar antes y después")
    largo = pd.DataFrame({"Antes": crudo[var], "Después": limpio[var]}).melt(var_name="etapa", value_name="valor")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.boxplot(data=largo, x="etapa", y="valor", hue="etapa", palette={"Antes": AMB, "Después": MENU[1]}, legend=False, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel(an.ETIQ[var]); ax.set_title("Antes y después de la limpieza")
    mostrar(fig)
    r0, r1 = an.regresion_simple(crudo.dropna(), "horas_estudio_semana"), an.regresion_simple(limpio, "horas_estudio_semana")
    c1, c2, c3 = st.columns(3)
    with c1:
        tarjeta("r estudio vs calificación (antes)", f"{r0['r']:.3f}", MENU[3])
    with c2:
        tarjeta("r estudio vs calificación (después)", f"{r1['r']:.3f}", MENU[1])
    with c3:
        tarjeta("Pendiente antes → después", f"{r0['pendiente']:.2f} → {r1['pendiente']:.2f}", MENU[5])
    st.write("La limpieza **cambia el resultado**: los errores de captura debilitaban la relación entre horas de estudio y calificación. "
             "Sin limpiar, se habría subestimado el efecto del estudio.")
    z = pd.DataFrame(an.StandardScaler().fit_transform(limpio[an.FEATS]), columns=an.FEATS).agg(["mean", "std"]).round(3)
    st.write("**Comprobación del z-score (media 0 y desviación 1):**")
    st.dataframe(z)


def p_inferencial():
    encabezado("Una diferencia puede ser casualidad: la prueba t y la regresión ayudan a decidir si es real.")
    st.subheader("Prueba t: ¿el taller mejora la calificación?")
    st.write("**H₀:** μ(con taller) − μ(sin taller) ≤ 0 (el taller no mejora la calificación promedio).  \n"
             "**H₁:** μ(con taller) − μ(sin taller) > 0 (el taller sí la mejora).  \nPrueba t de Student de dos muestras independientes, α = 0.05.")
    r = an.prueba_t(limpio_f)
    if r is None:
        st.warning("Con este filtro no hay suficientes estudiantes en ambos grupos.")
    else:
        for col, (t, v, c) in zip(st.columns(5), [("Media con taller", f"{r['media_si']:.2f}", MENU[1]), ("Media sin taller", f"{r['media_no']:.2f}", MENU[2]),
                                                  ("Estadístico t", f"{r['t']:.3f}", MENU[3]), ("Valor p", fmt_p(r["p"]), MENU[4]),
                                                  ("IC 95% de la diferencia", f"[{r['ic'][0]:.2f}, {r['ic'][1]:.2f}]", MENU[1])]):
            with col:
                tarjeta(t, v, c)
        st.write("")
        if r["p"] < 0.05:
            st.success(f"Se rechaza H₀ (p < 0.05): quienes tomaron el taller tienen en promedio {r['dif']:.2f} puntos más. "
                       f"Con 95% de confianza, la mejora está entre {r['ic'][0]:.2f} y {r['ic'][1]:.2f} puntos.")
        else:
            st.warning("No se rechaza H₀: no hay evidencia suficiente de que el taller mejore la calificación.")
        st.caption(f"Supuesto de varianzas iguales (Levene): p = {r['levene_p']:.3f}. Limitación: en un caso real la participación en el taller "
                   "no se asigna al azar, así que podrían influir otros factores.")
    st.subheader("Regresión lineal simple y correlación de Pearson")
    x = elegir("v_reg", [c for c in an.VARS if c != "calificacion"], "Variable predictora de la calificación")
    g = an.regresion_simple(limpio_f, x)
    fig, ax = plt.subplots(figsize=(7, 4))
    g_reg(limpio_f, x, ax)
    mostrar(fig)
    for col, (t, v, c) in zip(st.columns(4), [("Pearson r", f"{g['r']:.3f}", MENU[1]), ("R²", f"{g['r2'] * 100:.1f}%", MENU[3]),
                                              ("Pendiente", f"{g['pendiente']:.3f}", MENU[2]), ("Valor p", fmt_p(g["p"]), MENU[4])]):
        with col:
            tarjeta(t, v, c)
    st.write("")
    sentido = "aumenta" if g["pendiente"] > 0 else "disminuye"
    st.write(f"Por cada unidad adicional en **{an.ETIQ[x].lower()}**, la calificación {sentido} en promedio {abs(g['pendiente']):.2f} puntos; "
             f"la variable explica {g['r2'] * 100:.1f}% de la variabilidad de la calificación.")
    st.write(f"**Supuestos y limitaciones:** normalidad de residuos (Shapiro-Wilk): p = {g['shapiro_p']:.3f} "
             f"({'se cumple' if g['shapiro_p'] > 0.05 else 'no se cumple del todo, así que se interpreta con cautela'}). "
             "Además, **correlación no implica causalidad**: que dos variables se muevan juntas no demuestra que una cause a la otra.")


def p_kmeans():
    encabezado("K-means agrupa a los estudiantes por su parecido; no pone etiquetas clínicas, solo describe hábitos.")
    st.subheader("Segmentación con K-means")
    st.write("Variables usadas (estandarizadas): minutos de atención, horas de estudio y horas de sueño.")
    sil = an.silueta(limpio_f)
    k_opt = max(sil, key=sil.get)
    a, b = st.columns([1, 2])
    with a:
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.plot(list(sil), list(sil.values()), marker="o", color=MENU[1])
        ax.scatter([k_opt], [sil[k_opt]], s=160, color=ROJ, zorder=5)
        ax.set_xlabel("k"); ax.set_ylabel("Silueta promedio"); ax.set_title("Elección de k")
        mostrar(fig)
    k = st.slider("Número de segmentos (k)", 2, 6, k_opt)
    seg, perfil, cent = an.segmentar(limpio_f, k)
    nivel = "fuerte" if sil[k_opt] > 0.5 else "razonable" if sil[k_opt] > 0.25 else "débil"
    st.write(f"La silueta es máxima con **k = {k_opt}** ({sil[k_opt]:.3f}, estructura {nivel}); por eso se propone ese número de grupos. "
             f"Con k = {k} la silueta es {sil[k]:.3f}.")
    colores = {s: COL_SEG.get(s, c) for s, c in zip(perfil["segmento"], [MENU[1], AMB, ROJ, MENU[3], MENU[5], "#8D99AE"])}
    with b:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        g_seg(seg, cent, colores, ax)
        mostrar(fig)
    st.write("**Perfil y acción recomendada por segmento:**")
    tabla = perfil.rename(columns={"segmento": "Segmento", "estudiantes": "Estudiantes", "atencion": "Atención (min)", "estudio": "Estudio (h/sem)",
                                   "sueno": "Sueño (h)", "celular": "Celular (h/día)", "calificacion": "Calificación", "accion": "Acción recomendada"})
    st.dataframe(tabla[["Segmento", "Estudiantes", "Atención (min)", "Estudio (h/sem)", "Sueño (h)", "Celular (h/día)", "Calificación",
                        "Acción recomendada"]], hide_index=True)
    st.caption("Los segmentos describen hábitos de estudio; no son diagnósticos ni etiquetas de las personas.")


def p_dashboard():
    encabezado("Todo en un vistazo: elige un grupo y una variable y revisa qué apoyo conviene a cada perfil.")
    st.subheader("Dashboard")
    c1, c2 = st.columns(2)
    grupo = c1.radio("Grupo", ["Todos", "Con taller", "Sin taller"], horizontal=True)
    x = c2.selectbox("Variable predictora de la calificación", [c for c in an.VARS if c != "calificacion"],
                     format_func=lambda c: an.ETIQ[c], key="v_dash")
    d = limpio_f if grupo == "Todos" else limpio_f[limpio_f.taller_gestion_tiempo == ("Sí" if grupo == "Con taller" else "No")]
    if len(d) < 20:
        st.warning("Hay muy pocos estudiantes con esa combinación de filtros.")
        return
    seg3, perfil3, cent3 = an.segmentar(limpio_f, 3)
    seg_d = seg3.loc[d.index]
    pct = (seg_d["segmento"] == "Dispersos con poco sueño").mean()
    for col, (t, v, c) in zip(st.columns(4), [("Estudiantes", f"{len(d)}", MENU[1]), ("Calificación promedio", f"{d['calificacion'].mean():.1f}", MENU[2]),
                                              ("Atención promedio", f"{d['minutos_atencion'].mean():.1f} min", MENU[3]),
                                              ("En el perfil de distracción con poco sueño", f"{pct:.0%}", MENU[5])]):
        with col:
            tarjeta(t, v, c)
    st.write("")
    colores = {s: COL_SEG.get(s, MENU[1]) for s in perfil3["segmento"]}
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 3.8)); g_hist(d, "minutos_atencion", ax); mostrar(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 3.8)); g_reg(d, x, ax); mostrar(fig)
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots(figsize=(6, 3.8)); g_caja(limpio_f, "calificacion", ax); mostrar(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 3.8)); g_seg(seg_d, cent3, colores, ax); mostrar(fig)
    r = an.prueba_t(limpio_f)
    preds = [c for c in an.VARS if c != "calificacion"]
    rs = {c: an.regresion_simple(limpio_f, c)["r"] for c in preds}
    top = max(rs, key=lambda c: abs(rs[c]))
    st.markdown("### Conclusión")
    st.markdown(f"""
1. Después de limpiar, el factor más asociado con la calificación es **{an.ETIQ[top].lower()}** (r = {rs[top]:.2f}).
2. {("El taller de gestión del tiempo se asocia con una mejora de **" + f"{r['dif']:.1f}" + " puntos** (p " + ("< 0.05" if r["p"] < 0.05 else ">= 0.05") + ").") if r else "No hay suficientes datos para comparar el taller con este filtro."}
3. Existen **tres perfiles** de regulación de la atención con necesidades distintas, no un solo tipo de estudiante.""")
    st.markdown("### Recomendación para el centro de orientación")
    for _, f in perfil3.sort_values("calificacion").iterrows():
        st.markdown(f"- **{f['segmento']}** ({int(f['estudiantes'])} estudiantes, calificación promedio {f['calificacion']:.1f}): {f['accion']}")
    st.markdown("""**Prioridad sugerida:** empezar por el perfil con menor calificación promedio y ampliar el taller de gestión del tiempo, que mostró una mejora medible.

**Límites y ética:** los datos son simulados y las relaciones son asociaciones, no causas. Los perfiles orientan el apoyo, pero la decisión sobre cada persona
debe tomarla un profesional, con su consentimiento y sin diagnosticar ni sancionar.""")


{"inicio": p_inicio, "situacion": p_situacion, "descriptiva": p_descriptiva, "limpieza": p_limpieza,
 "inferencial": p_inferencial, "kmeans": p_kmeans, "dashboard": p_dashboard}[pag]()

st.markdown('<div class="pie"><span>Proyecto-TDAH</span><span>Instituto Tecnológico de Cuautla</span><span>Gil Toledano Nicol Abigail</span></div>',
            unsafe_allow_html=True)
