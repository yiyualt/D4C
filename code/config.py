task_configs = {
    1: {
        "auxiliary": "AM_Electronics",
        "target": "AM_CDs",
        "lr": 5e-4,
        "coef": 1,
        "adv": 0.01
    },
    2: {
        "auxiliary": "AM_Movies",
        "target": "AM_CDs",
        "lr": 1e-3,
        "coef": 0.1,
        "adv": 0.01
    },
   3: {
        "auxiliary": "AM_CDs",
        "target": "AM_Electronics",
        "lr": 5e-4,
        "coef": 0.5,
        "adv": 0.1
    },
   4: {
        "auxiliary": "AM_Movies",
        "target": "AM_Electronics",
        "lr": 1e-3,
        "coef": 0.5,
        "adv": 0.01
    },
   5: {
        "auxiliary": "AM_CDs",
        "target": "AM_Movies",
        "lr": 1e-3,
        "coef": 0.5,
        "adv": 0.01
    },
   6: {
        "auxiliary": "AM_Electronics",
        "target": "AM_Movies",
        "lr": 1e-3,
        "coef": 0.5,
        "adv": 0.01
    },
    7: {
        "auxiliary": "Yelp",
        "target": "TripAdvisor",
        "lr": 1e-4,
        "coef": 0.5,
        "adv": 0.01
    },
    8: {
        "auxiliary": "TripAdvisor",
        "target": "Yelp",
        "lr": 5e-4,
        "coef": 1,
        "adv": 0.01
    }
}

def get_task_config(task_idx):
    return task_configs.get(task_idx, None)
