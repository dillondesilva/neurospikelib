from lif_new import LIFSimulation

LIFSimulation.simulate(pulses=[{
    "start": 20,
    "end": 30,
    "amp": 10
}, {
    "start": 40,
    "end": 50,
    "amp": 20   
}], resolution=1)