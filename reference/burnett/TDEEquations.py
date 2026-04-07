import numpy as np
import ThermoMlReader as tml

class EqWrapper:
    @staticmethod
    def run(VarsRange, Params):
        pass

class ViscosityL(EqWrapper):
    "Liquid-Gas Equilibrium viscosity"
    @staticmethod
    def run(T, Params):

        
        #add zeros to params if params is not 4
        missing = 4 - len(Params)
        A, B, C, D = Params + [0] * missing
        
        y = np.exp(A + (B / T) + (C / T**2) + (D / T**3))
        return y

class PPDS9(EqWrapper):
    "Saturated Liquid viscosity"
    @staticmethod
    def run(T, Params):
        a5, a1, a2, a3, a4 = Params
        
        # Calculate X based on the equation given
        X = (a3 - T) / (T - a4)
        
        # Calculate the natural logarithm of the viscosity ratio
        ln_viscosity_ratio = a1 * X**(1/3) + a2 * X**(4/3) + np.log(a5)
        
        # Since ho is 1 Pa×s, we can directly exponentiate the ln_viscosity_ratio to get the viscosity
        viscosity = np.exp(ln_viscosity_ratio)
        
        return viscosity

class PPDS14(EqWrapper):
    "Surface tension liquid-gas"
    @staticmethod
    def run(T, Params):
        Tc = Params[-1]  # Assuming the last parameter is Tc
        t = 1 - T / Tc
        a0, a1, a2 = Params[:-1]  # Exclude the last parameter Tc
        y = a0 * np.power(t, a1) * (1 + a2 * t)
        return y,Tc


class Watson(EqWrapper):
    "Watson Equation for specific property calculation"
    @staticmethod
    def run(T, Params):


        # Ensuring T does not reach or exceed Tc
        Tc = Params[-1]
        #T = T[T < Tc]

        a1 = Params[0]
        ai_coeffs = Params[1:-1]  # Exclude Tc

        Tr = T / Tc

        # Calculation of ln(s/so)
        sum_terms = np.array([ai * np.power(Tr, i) * np.log(1 - Tr) for i, ai in enumerate(ai_coeffs, start=1)])
        ln_s_so = a1 + np.sum(sum_terms, axis=0)
        #print when nans are present
        if np.isnan(ln_s_so).any():
            print('nans present')
        s_s0 = np.exp(ln_s_so)
        return s_s0,Tc
#s/so = åai ×(1 - T/Tc)i, 
class ISETExpansion(EqWrapper):
    "ISET Expansion for specific property calculation"
    @staticmethod
    def run(T, Params):
        ai_coeffs = Params[1:]
        Tc = Params[0]  # Exclude Tc
        s_s0 = sum([ai * (1 - T / Tc)**i for i, ai in enumerate(ai_coeffs, start=1)])
        return s_s0,Tc