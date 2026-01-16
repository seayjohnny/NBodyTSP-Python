
# att48
# -----
# N-body path cost: 35799.9961
# Optimal cost: 33523.7070
# Percent error: 6.79%
att48 = {
    "simulator_options": {},
    "nbody_options": {
        "slope_repulsion": 50.0,
        "mag_attraction": 25,
        "force_cutoff_extra": 100,
        "WALL_STRENGTH": 20000.0,
        "MASS": 80,
        "DAMP": 20.0,
    }
}

# ch150
# -----
# N-body path cost: 7634.9990
# Optimal cost: 6532.2798
# Percent error: 16.88%
ch150 = {
    "simulator_options": {
        "use_pressure": True,
    },
    "nbody_options": {
        "slope_repulsion": 50.0,
        "mag_attraction": 25,
        "force_cutoff_extra": 100,
        "WALL_STRENGTH": 200.0,
        "MASS": 80,
        "DAMP": 20.0,
    }
}