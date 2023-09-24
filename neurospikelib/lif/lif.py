from neuron import h
from neuron.units import mV, ms


class LIFSimulation:
    '''
    Leaky Integrate and Fire Simulation Module for Neurospike. Operates
    as a wrapper around brian2 and neurodynex3 libraries to provide
    LIF Model support for Neurospike simulation app
    '''
    def __init__(self, stimulation_parameters, neuron_parameters, simulation_duration):
        '''Instantiates a new LIF Simulation with given parameters'''
        self._stimulation_parameters = stimulation_parameters
        self._neuron_parameters = neuron_parameters
        self._simulation_duration = simulation_duration

    def simulate(self, input_type="linear"):
        '''Runs LIF model simulation by given parameters'''
        # Create soma-only neuron model with passive
        # leak channels for membrane
        soma = h.Section(name="soma")
        soma.insert("pas")
        soma.L = self._neuron_parameters.length
        soma.diam = self._neuron_parameters.diam

        # Create step current from input parameters
        stim = h.IClamp(soma(0.5))
        stim.dur = self._stimulation_parameters.duration
        stim.amp = self._stimulation_parameters.amplitude
        soma_v = h.Vector().record(soma(0.5)._ref_v)
        time = h.Vector().record(h._ref_t)

        # Run simulation
        resting_membrane_potential = self._neuron_parameters.resting_v 
        simulation_duration = self._simulation_duration * ms
        h.finitialize(-70 * mV)
        h.continuerun(simulation_duration)

        simulation_results = {
            "membrane_voltage": soma_v,
            "timepoints": time
        }

        return simulation_results
        