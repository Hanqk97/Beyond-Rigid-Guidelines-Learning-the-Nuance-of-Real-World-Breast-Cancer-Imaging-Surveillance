# stage_utils.py

def normalize_t_stage(t_value):
    """
    Normalize raw T stage into coarse categories:
    T0, Tis, T1, T2, T3, T4, TX
    """
    if t_value is None:
        return None

    t = str(t_value).strip()

    if t == "" or t.lower() in {"nan", "none", "null", "invalid", "unknown", "not applicable"}:
        return None

    t = t.upper()

    if "TIS" in t:
        return "Tis"
    if "TX" in t:
        return "TX"
    if "T0" in t:
        return "T0"
    if "T1" in t:
        return "T1"
    if "T2" in t:
        return "T2"
    if "T3" in t:
        return "T3"
    if "T4" in t:
        return "T4"

    return None


def normalize_n_stage(n_value):
    """
    Normalize raw N stage into coarse categories:
    N0, N1mi, N1, N2, N3, NX
    """
    if n_value is None:
        return None

    n = str(n_value).strip()

    if n == "" or n.lower() in {"nan", "none", "null", "invalid", "unknown", "not applicable"}:
        return None

    n = n.upper()

    if "NX" in n:
        return "NX"
    if "N1MI" in n:
        return "N1mi"
    if "N3" in n:
        return "N3"
    if "N2" in n:
        return "N2"
    if "N1" in n:
        return "N1"
    if "N0" in n:
        return "N0"

    return None


def normalize_m_stage(m_value):
    """
    Normalize raw M stage into coarse categories:
    M0, M1, MX
    """
    if m_value is None:
        return None

    m = str(m_value).strip()

    if m == "" or m.lower() in {"nan", "none", "null", "invalid", "unknown", "not applicable"}:
        return None

    m = m.upper()

    if "MX" in m:
        return "MX"
    if "M1" in m:
        return "M1"
    if "M0" in m:
        return "M0"

    return None


def tnm_to_stage_anatomic(T, N, M, keep_stage0=False, detailed=False):
    """
    Map coarse TNM categories to breast anatomic stage.

    Clinical mapping used:
    Stage 0:   Tis N0 M0
    Stage IA:  T1 N0 M0
    Stage IB:  T0/T1 N1mi M0
    Stage IIA: T0/T1 N1 M0 or T2 N0 M0
    Stage IIB: T2 N1 M0 or T3 N0 M0
    Stage IIIA: T0/T1/T2 N2 M0 or T3 N1/N2 M0
    Stage IIIB: T4 N0/N1/N2 M0
    Stage IIIC: Any T N3 M0
    Stage IV: Any T Any N M1

    Parameters
    ----------
    T, N, M : str or None
        Already-normalized coarse TNM categories.
    keep_stage0 : bool
        If False, Stage 0 returns None.
    detailed : bool
        If True, return labels like 'IA', 'IIIB'.
        If False, return collapsed numeric stage 0/1/2/3/4.

    Returns
    -------
    str or int or None
    """
    if T is None or N is None or M is None:
        return None

    T = str(T).strip()
    N = str(N).strip()
    M = str(M).strip()

    if T == "TX" or N == "NX" or M == "MX":
        return None

    if M == "M1":
        return "IV" if detailed else 4

    if M != "M0":
        return None

    stage = None

    if T == "Tis" and N == "N0":
        stage = "0"
    elif T in {"T0", "T1"} and N == "N1mi":
        stage = "IB"
    elif T == "T1" and N == "N0":
        stage = "IA"
    elif ((T in {"T0", "T1"} and N == "N1") or
          (T == "T2" and N == "N0")):
        stage = "IIA"
    elif ((T == "T2" and N == "N1") or
          (T == "T3" and N == "N0")):
        stage = "IIB"
    elif ((T in {"T0", "T1", "T2"} and N == "N2") or
          (T == "T3" and N in {"N1", "N2"})):
        stage = "IIIA"
    elif T == "T4" and N in {"N0", "N1", "N2"}:
        stage = "IIIB"
    elif N == "N3":
        stage = "IIIC"

    if stage is None:
        return None

    if detailed:
        if stage == "0" and not keep_stage0:
            return None
        return stage

    stage_to_num = {
        "0": 0,
        "IA": 1,
        "IB": 1,
        "IIA": 2,
        "IIB": 2,
        "IIIA": 3,
        "IIIB": 3,
        "IIIC": 3,
        "IV": 4,
    }

    out = stage_to_num[stage]
    if out == 0 and not keep_stage0:
        return None
    return out


def raw_tnm_to_stage(t_raw, n_raw, m_raw, keep_stage0=False, detailed=False):
    """
    Full pipeline:
    raw T/N/M -> normalize -> anatomic stage
    """
    T = normalize_t_stage(t_raw)
    N = normalize_n_stage(n_raw)
    M = normalize_m_stage(m_raw)

    return tnm_to_stage_anatomic(
        T=T,
        N=N,
        M=M,
        keep_stage0=keep_stage0,
        detailed=detailed
    )