RB_MANUAL = {
    # model_name: { norm: robust_accuracy_percent }
    "Carmon2019Unlabeled": {
        "linf": 59.53,  
        "l2":   0.0,  
    },
    "Rice2020Overfitting": {
        "linf": 53.42,
        "l2":   67.68,
    },
    "Rade2021Helper_extra": {
        "linf": 62.83,
        "l2":   0.0,
    },
    "Bartoldson2024Adversarial_WRN-94-16": {
        "linf": 73.71,
        "l2":   0.0,
    },
    "Kang2021Stable": {
        "linf": 64.20,
        "l2":   0.0,
    },
    "Amini2024MeanSparse_Ra_WRN_70_16": {
        "linf": 68.94,
        "l2":   0.0,
    },
    "Gowal2021Improving_70_16_ddpm_100m": {
        "linf": 66.10,
        "l2":   0.0,
    } 
}

def get_robustbench_point(model_name: str, p_model: str):
    if p_model == "linf":
        threat_model = "Linf"
        rb_epsilon = 8.0 / 255.0
    elif p_model == "l2":
        threat_model = "L2"
        rb_epsilon = 0.5
    else:
        return None  

    model_entry = RB_MANUAL.get(model_name)
    if not model_entry:
        return None

    rb_rob = model_entry.get(p_model)
    if rb_rob is None:
        return None

    return {
        "epsilon": float(rb_epsilon),
        "robust_accuracy": float(rb_rob),  # already in percent
        "threat_model": threat_model,
        "source": "manual"
  }
