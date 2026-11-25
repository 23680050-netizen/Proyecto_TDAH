"""Datos y análisis del proyecto final (sin Streamlit, para poder probarlos aparte)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEMILLA = 42
VARS = ["horas_estudio_semana", "horas_sueno", "minutos_atencion", "uso_celular_horas", "calificacion"]
ETIQ = {"horas_estudio_semana": "Horas de estudio por semana", "horas_sueno": "Horas de sueño por noche",
        "minutos_atencion": "Minutos de atención sostenida", "uso_celular_horas": "Uso de celular al día (h)",
        "calificacion": "Calificación promedio (0 a 100)"}
FEATS = ["minutos_atencion", "horas_estudio_semana", "horas_sueno"]
ACCIONES = {
    "Enfocados constantes": "Mantener su ritmo: invitarlos como tutores pares y reconocer su constancia.",
    "Esforzados con atención corta": "Taller de técnicas de estudio con pausas y tutoría de manejo del tiempo; vigilar el desgaste.",
    "Dispersos con poco sueño": "Taller de higiene del sueño y uso consciente del celular; invitación voluntaria a una valoración con profesionales de la salud.",
}


# ------------------------------------------------------------ datos propios (simulados, con defectos a propósito)
def generar_datos():
    rng = np.random.default_rng(SEMILLA)
    # (n, atención, estudio, sueño, celular) -> (media, desviación)
    perfiles = [(120, (38, 6), (14, 3), (7.2, 0.7), (2.5, 0.8)),
                (100, (16, 4), (8, 2.5), (5.3, 0.8), (5.5, 1.2)),
                (80, (22, 5), (20, 3), (6.0, 0.8), (3.5, 1.0))]
    bloques = [pd.DataFrame({"minutos_atencion": rng.normal(*a, n), "horas_estudio_semana": rng.normal(*e, n),
                             "horas_sueno": rng.normal(*s, n), "uso_celular_horas": rng.normal(*c, n)})
               for n, a, e, s, c in perfiles]
    df = pd.concat(bloques, ignore_index=True).clip(lower=0.5)
    n = len(df)
    df["taller_gestion_tiempo"] = np.where(rng.random(n) < 0.35, "Sí", "No")
    df["semestre"] = rng.choice(np.arange(2, 10), n)
    df["calificacion"] = (45 + 1.0 * df.horas_estudio_semana + 0.45 * df.minutos_atencion + 1.2 * df.horas_sueno
                          - 0.8 * df.uso_celular_horas + 4 * (df.taller_gestion_tiempo == "Sí")
                          + rng.normal(0, 4, n)).clip(30, 100)
    df[VARS] = df[VARS].round(1)
    df = df.sample(frac=1, random_state=SEMILLA).reset_index(drop=True)
    df.insert(0, "id_estudiante", [f"E{i:03d}" for i in range(1, n + 1)])
    # Problemas de calidad introducidos con intención
    idx = rng.permutation(n)
    df.loc[idx[:27], "horas_sueno"] = np.nan                                      # 9% faltantes
    df.loc[idx[27:45], "minutos_atencion"] = np.nan                               # 6% faltantes
    df.loc[idx[45:53], "uso_celular_horas"] = rng.uniform(15, 22, 8).round(1)     # errores de captura
    df.loc[idx[53:58], "horas_estudio_semana"] = rng.uniform(55, 80, 5).round(1)  # horas del mes, no de la semana
    df.loc[idx[58:61], "calificacion"] = [108.0, 115.0, 120.0]                    # fuera de la escala 0 a 100
    return df


# ------------------------------------------------------------ U2: descriptiva y regla de Tukey
def resumen(df, cols=VARS):
    d = df[cols]
    t = pd.DataFrame({"n": d.count(), "faltantes": d.isna().sum(), "media": d.mean(), "mediana": d.median(),
                      "desv_est": d.std(), "mínimo": d.min(), "máximo": d.max()})
    t["rango"] = t["máximo"] - t["mínimo"]
    t.index = [ETIQ[c] for c in t.index]
    return t.round(2)


def limites_tukey(s):
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1, q3, iqr, q1 - 1.5 * iqr, q3 + 1.5 * iqr


def atipicos(df, cols=VARS):
    filas = []
    for c in cols:
        q1, q3, iqr, li, ls = limites_tukey(df[c])
        filas.append({"variable": ETIQ[c], "Q1": q1, "Q3": q3, "IQR": iqr, "límite inferior": li,
                      "límite superior": ls, "atípicos": int(((df[c] < li) | (df[c] > ls)).sum())})
    return pd.DataFrame(filas).set_index("variable").round(2)


# ------------------------------------------------------------ U3: limpieza
def limpiar(df):
    d = df.copy()
    limites = {c: limites_tukey(df[c])[3:] for c in VARS}      # límites calculados con los datos originales
    for c in VARS:
        d[c] = d[c].fillna(d[c].median())                        # imputación por mediana
        d[c] = d[c].clip(max(limites[c][0], 0), limites[c][1])   # capping con los límites de Tukey
    d["calificacion"] = d["calificacion"].clip(upper=100)        # la escala máxima es 100
    return d.round(2)


# ------------------------------------------------------------ U4: inferencia
def prueba_t(d):
    a = d.loc[d.taller_gestion_tiempo == "Sí", "calificacion"]
    b = d.loc[d.taller_gestion_tiempo == "No", "calificacion"]
    if len(a) < 2 or len(b) < 2:
        return None
    t, p = stats.ttest_ind(a, b, alternative="greater")          # H1: mu_taller - mu_sin > 0
    gl = len(a) + len(b) - 2
    sp2 = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / gl
    ee = np.sqrt(sp2 * (1 / len(a) + 1 / len(b)))
    dif, tc = a.mean() - b.mean(), stats.t.ppf(0.975, gl)
    return dict(media_si=a.mean(), media_no=b.mean(), n_si=len(a), n_no=len(b), t=t, p=p, gl=gl, dif=dif,
                ic=(dif - tc * ee, dif + tc * ee), levene_p=stats.levene(a, b).pvalue)


def regresion_simple(d, x):
    r = stats.linregress(d[x], d["calificacion"])
    res = d["calificacion"] - (r.intercept + r.slope * d[x])
    return dict(r=r.rvalue, p=r.pvalue, pendiente=r.slope, intercepto=r.intercept, r2=r.rvalue ** 2,
                shapiro_p=stats.shapiro(res).pvalue, residuos=res)


# ------------------------------------------------------------ U5: K-means
def silueta(d, ks=range(2, 7)):
    X = StandardScaler().fit_transform(d[FEATS])
    return {k: silhouette_score(X, KMeans(k, random_state=SEMILLA, n_init=10).fit_predict(X)) for k in ks}


def segmentar(d, k):
    sc = StandardScaler()
    X = sc.fit_transform(d[FEATS])
    km = KMeans(k, random_state=SEMILLA, n_init=10).fit(X)
    e = d.copy()
    e["cluster"] = km.labels_
    perfil = e.groupby("cluster").agg(estudiantes=("id_estudiante", "count"), atencion=("minutos_atencion", "mean"),
                                      estudio=("horas_estudio_semana", "mean"), sueno=("horas_sueno", "mean"),
                                      celular=("uso_celular_horas", "mean"), calificacion=("calificacion", "mean")).round(2)
    if k == 3:
        c_enf = perfil["atencion"].idxmax()
        c_esf = perfil.drop(c_enf)["estudio"].idxmax()
        c_dis = [c for c in perfil.index if c not in (c_enf, c_esf)][0]
        nombres = {c_enf: "Enfocados constantes", c_esf: "Esforzados con atención corta", c_dis: "Dispersos con poco sueño"}
    else:
        nombres = {c: f"Segmento {c + 1}" for c in perfil.index}
    perfil["segmento"] = perfil.index.map(nombres)
    perfil["accion"] = perfil["segmento"].map(ACCIONES).fillna("Definir el apoyo según el comportamiento promedio del segmento.")
    e["segmento"] = e["cluster"].map(nombres)
    cent = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=FEATS)
    cent["segmento"] = cent.index.map(nombres)
    return e, perfil, cent
