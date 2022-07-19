from math import pi

defaults = {
    "B0": 1.0,
    "R0": 239.8081535,
    "rMin": 0.1,
    "rMax": 14.5,
    "zMin": 0.0,
    "vMax": 7.32,
    "eps": 1e-6,
    "eps0": 8.854187817e-12,
    "kN0": 0.055,
    "kTi": 0.27586,
    "deltaRTi": 1.45,
    "CTi": 1.0,
    "m": 15,
    "n": 1,
    "iotaVal": 0.0,
    "npts": [256, 512, 32, 128],
    "splineDegrees": [3, 3, 3, 3],
    "dt": 2
}

defaults["vMin"] = -defaults["vMax"]
defaults["zMax"] = defaults["R0"]*2*pi
defaults["deltaRTe"] = defaults["deltaRTi"]
defaults["kTe"] = defaults["kTi"]
defaults["deltaRN0"] = 2.0*defaults["deltaRTe"]
defaults["deltaR"] = 4.0*defaults["deltaRN0"]/defaults["deltaRTi"]
defaults["CTe"] = defaults["CTi"]
