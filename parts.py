import os
from pathlib import Path
import pandas as pd
import numpy as np
import sympy as sp
from sympy.physics.continuum_mechanics.beam import Beam
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import griddata, CloughTocher2DInterpolator, NearestNDInterpolator
import pprint
from IPython.display import clear_output
import logging

plt.rcParams["figure.dpi"] = 125

prog_path = Path(os.path.abspath(""))
table_path = prog_path / "Tables"
out_table_path = prog_path / "Output_Tables"
fig_path = prog_path / "Figures"

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

logging.basicConfig(
    filename="app.log", 
    filemode="w", # "a" for append (default), "w" to overwrite
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

handler = logging.FileHandler("app.log", mode="w")

logging.info("Logging started")

def export_latex_table(df: pd.DataFrame, file_name: str) -> None:
    with open(out_table_path / f"{file_name}.tex", "w") as f:
        res = df.to_latex(
            index=False, 
            # float_format="%.3f", 
            column_format="|c"*df.shape[1] + "|"
        )
        res = res.replace(r"\\" + "\n", r"\\ \hline" + "\n")
        res = res.replace(r"\toprule", r"\hline")
        res = res.replace(r"\midrule", "")
        res = res.replace(r"\bottomrule", "")
        f.write(res)

def format_value(x):
    if isinstance(x, (int, np.integer)):
        return str(x)
    elif isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "---"
        if np.isinf(x):
            return r"$\infty$"
        if x == int(x):
            return str(int(x))
        return f"{x:.3f}"
    return str(x)

# could add notes column to this
var_unit_map_df = pd.read_csv(table_path / "var_unit_map.csv").set_index("Python_Var")
var_unit_map = var_unit_map_df.to_dict(orient="index")

def get_instance_variables_df(obj):
    """
    creates a dataframe of instance variables, filtering out instances of other design classes and complex objects
    """
    custom_classes = (Material, Gear, GearTrain, RetainingRing, Key, Bearing, Shaft, Gearbox)

    data = []

    def process_dict(d):
        for k, v in d.items():
            if isinstance(v, custom_classes) or isinstance(v, (dict, list, tuple, np.ndarray, sp.Basic)) or k.startswith('_') or callable(v):
                continue
            
            if k in var_unit_map:
                latex_name = var_unit_map[k]["LaTeX_Var"]
                unit = var_unit_map[k]["Unit"]

                if unit == "" or unit == "N/A" or pd.isna(unit):
                    unit = "---"

                if k == "min_fos_key":
                    assert isinstance(v, str)
                    v = v.replace("_", "\_")
                
                data.append([latex_name, v, unit])
            else:
                print(f"Check if this export variable is intentionally missing: {k}")

    # process the main object attributes
    obj_vars = vars(obj)
    process_dict(obj_vars)

    # check for nested Material objects
    for v in obj_vars.values():
        if isinstance(v, Material):
            process_dict(vars(v))

    return pd.DataFrame(data, columns=["Variable", "Value", "Unit"])

## ----------------------------------------------------------
## SETUP ABOVE
## GEARBOX DESIGN BELOW
## ----------------------------------------------------------

n_in = 10_000 # motor rpm
n_out = 500 # wheel rpm
m_V = n_in // n_out # total velocity ratio
L_intermediate = 12 * 0.0254 # m
P = 50 * 745.699872 # Watts

# https://www.jjamusements.com/wp-content/uploads/2014/10/Manual-Go-Kart-20143.pdf
# GO-KART SERVICE SCHEDULE (Electric) p. 0-11 
# longest service schedule 500 hrs. bearing 100 hrs
L_desired_hrs = 1000 # hrs
L_desired_min = L_desired_hrs * 60 # min

n_target = 1.1 # target for minimum factor of safety

global_param_df = pd.DataFrame(
    [[r"$n_{in}$",          n_in,           "rpm"],
    [r"$n_{out}$",          n_out,          "rpm"],
    [r"$m_V$",              m_V,            "N/A"],
    [r"$L_{intermediate}$", L_intermediate, "m"],
    ["$P$",                 P,              "W"],
    [r"$n_{target}$",       n_target,       "N/A"],
    ["$L_D$",               L_desired_hrs,  "hr"]],
    columns=["Variable", "Value", "Unit"]
)

global_param_df["Value"] = global_param_df["Value"].apply(format_value)
export_latex_table(global_param_df, "system_parameters")

class TableInterpolator:
    """to be used for stress concentration factors in Table A-15"""
    def __init__(self, table_name):
        self.table_name = table_name
        self.df = pd.read_csv(table_path / f"{self.table_name}.csv")
        self.x_name, self.line_name, self.y_name = self.df.columns

        points = self.df[[self.x_name, self.line_name]].values
        values = self.df[self.y_name].values
        # gives nan when out of data hull
        self.interp = CloughTocher2DInterpolator(points, values, rescale=True)

        self.backup_interp = NearestNDInterpolator(points, values)

    def robust_interp(self, points):
        """tries interpolator. Uses nearest sample point when out of data hull"""
        pts = np.atleast_2d(points)
        vals = self.interp(pts)

        # find NaNs (outside convex hull)
        mask = np.isnan(vals)
        if np.any(mask):
            vals[mask] = self.backup_interp(pts[mask])

        return vals if len(vals) > 1 else vals[0]

    def plot(self, is_log_x=False, extra_x=None, extra_y=None):
        line_values = np.unique(self.df[self.line_name])
        x_values = np.linspace(self.df[self.x_name].min(), self.df[self.x_name].max(), 200)

        fig, ax = plt.subplots(figsize=(7, 5))

        for line_val in line_values:
            query_points = np.column_stack((x_values, np.full_like(x_values, line_val)))
            y_values = self.interp(query_points)
            # y_values = self.robust_interp(query_points)
            
            line, = plt.plot(x_values, y_values, label=f"{line_val}")
            plt.text(
                x_values[-1]*1.025, y_values[-1], f"{line_val}", 
                va="center", fontsize=10, color=line.get_color()
            )

        plt.scatter(self.df[self.x_name], self.df[self.y_name], color="black", s=5, alpha=0.9, label="Original Points")

        if (extra_x is not None) and (extra_y is not None):
            plt.scatter(extra_x, extra_y)

        if is_log_x:
            ax.set_xscale("symlog")

        # ticks = np.np.linspace(df[self.x_name].min(), df[self.x_name].max(), 10)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(min_n_ticks=8))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

        # extended slightly for text labels
        plt.xlim(self.df[self.x_name].min()*0.9, self.df[self.x_name].max()*(1.25 if is_log_x else 1.1))

        plt.grid()
        plt.xlabel(self.x_name)
        plt.ylabel(self.y_name)
        plt.title(f"{self.table_name}, lines are each {self.line_name}")
        plt.tight_layout()
        plt.savefig(fig_path / f"{self.table_name}.png")

        return fig, ax

FIG_14_6 = TableInterpolator("Fig_14-6")
A_15_8 = TableInterpolator("A-15-8")
A_15_9 = TableInterpolator("A-15-9")
A_15_16 = TableInterpolator("A-15-16")
A_15_17 = TableInterpolator("A-15-17")

class Material:
    # maximum gearbox temp for automotive transmission: https://tanhon.com/what-is-the-temperature-range-of-a-gearbox/
    T_C = 115
    k_d = 0.99 + 5.9e-4 * T_C - 2.1e-6 * T_C**2

    reliability_factors = {0.50: 1.000, 0.90: 0.897, 0.95: 0.868, 0.99: 0.814, 0.999: 0.753, 0.9999: 0.702}
    reliability = 0.90
    
    k_e = reliability_factors[reliability]

    def __init__(self, name, S_ut, S_y, H_B):
        """S_ut (MPa), S_y (MPa), H_B (Brinell Hardness)"""

        self.name = name
        self.S_ut_MPa = S_ut
        self.S_y_MPa = S_y
        self.S_e_prime_MPa = 0.5 * S_ut if S_ut <= 1400.0 else 700.0  

        self.S_ut = S_ut * 1e6 # Pa
        self.S_y = S_y * 1e6 # Pa
        self.S_e_prime = self.S_e_prime_MPa * 1e6 # Pa
        self.H_B = H_B

        assert 340 <= S_ut <= 1700 # MPa, assumption for Eq 6-35 and Eq 6-36, if breaks then clip sqrt(a) with a function
        self.sqrt_a_bending = 1.24 - 2.25e-3*self.S_ut_MPa + 1.60e-6*self.S_ut_MPa**2 - 4.11e-10*self.S_ut_MPa**3
        self.sqrt_a_torsion = 0.958 - 1.83e-3*self.S_ut_MPa + 1.43e-6*self.S_ut_MPa**2 - 4.11e-10*self.S_ut_MPa**3

    def corrected_S_e(self, d):
        """
        only use with shaft
        d (mm)
        returns Pa
        """
        d_mm = d

        # Table 6-2, machined surface
        k_a = 3.04 * self.S_ut_MPa**(-0.217)
        
        # Eq 6-19
        if d_mm < 7.62:
            k_b = 1.0
        elif d_mm <= 51.0:
            k_b = (d_mm / 7.62)**(-0.107)
        elif d_mm <= 254.0:
            k_b = 1.51 * d_mm**(-0.157)
        else:
            k_b = 1.51 * 254.0**(-0.157)
            logging.info(f"shaft too big {d_mm = }")
        
        # rotating + bending
        k_c = 1

        return k_a * k_b * k_c * Material.k_d * Material.k_e * self.S_e_prime

# iter 1
generic_steel = Material("AISI 1045 CD", 630, 530, 179)

# iter 2
# shaft_steel = Material("AISI 1050 CD", 690, 580, 197)
shaft_steel = Material("AISI 4130 Q/T at 205 C", 1630, 1460, 467)

# old gear links: 
# https://us.c.misumi-ec.com/book/MSM1_USA_01/pdf/1107.pdf
# https://us.c.misumi-ec.com/book/MSM1_USA_01/pdf/1108.pdf
# https://us.c.misumi-ec.com/book/usa_2019_msm_fa/900/1540.jpg
# https://us.c.misumi-ec.com/book/usa_2019_msm_fa/900/1541.jpg

# GEAKBH1.0-80-8-A
# https://www.steelexpress.co.uk/steel-hardness-conversion.html
# 53 HRC -> 513 HB
# 55 HRC -> 552 HB
gear_steel = Material(r"\makecell{Induction\\Hardened\\AISI 1045}", 630, 530, 513)
# 570~750

class Gear:
    # Table 14-9. Electric motor is a very uniform power source. Go-kart tracks are smooth. However, go-karts encounter curb hits and heavy breaking. Light-moderate shock is used.
    # go-karts are most similar to car pullers in Table 29.1 of page 29.4 from https://iem.ca/pdf/resources/Standard%20Handbook%20of%20Machine%20Design.pdf
    K_o = 1.25

    Q_v = 7 # ok commercial quality
    
    # 14-27b
    B = 0.25*((12-Q_v)**(2/3))
    A = 50 + 56*(1-B)

    # 14-28
    V_t_max = ((A + Q_v - 3)**2) / 200

    df_14_2 = pd.read_csv(table_path / "Table_14-2.csv")

    # suggested value when more information is not known [p. 782, Shigley]
    K_s = 1.0
    
    # 14-31, uncrowned teeth
    C_mc = 1.0

    # 14-33, S_1∕S will likely be > 0.175
    C_pm = 1.1

    # Table 14-10, commercial enclosed unit
    A_14_10, B_14_10, C_14_10 = (0.127, 0.0158, -0.930e-4)

    # 14-35, gearing assembled normally
    C_e = 1.0

    # Fig 14-6, no rim
    K_B = 1.0

    # Sec. 14-17
    S_F = 1.0 
    S_H = 1.0

    # Fig 14-14, bending stress-cycle factor, general conservative value for high cycles
    # Y_N = 0.92
    # Fig 14-15, pitting stress-cycle factor, general conservative value for high cycles
    # Z_N = 0.87

    reliability = 0.90

    # Table 14-8, both pinion and gear are made of steel
    Z_E = 191 # sqrt(MPa)

    Z_R = 1 # Sec 14-9

    # Sec 14-12, assume hardness ratio between all pinions and gears are less than 1.2 
    Z_W = 1
    
    def __init__(
        self,
        material: Material, 
        N: int, # number of teeth
        module, # mm/teeth
        phi, # pressure angle (deg)
        F, # face width (mm)
        ID: float | None = None,
        number: int | None = None
    ):
        self.material = material
        self.N = N # number of teeth
        self.module = module # mm/teeth
        # self.diametral_pitch = 1 / self.module # need to convert units
        self.phi = phi # deg
        self.F = F # mm

        if (self.F < 3*np.pi*self.module) or (self.F > 5*np.pi*self.module):
            # print("atypical face width as compared to module")
            logging.info("atypical face width as compared to module")
        
        # nominal
        self.d = self.N * self.module # pitch diameter (mm)
        # self.p = np.pi * self.module # circular pitch (mm)
        self.ID = ID # nominal bore diameter (mm)
        # convert nominals to dimensions with tolerances later? mostly just ID so can compare with shaft

        # Table 13-3, assumes phi = 20 deg
        self.addendum = self.module
        self.dedendum = 1.25 * self.module

        # https://www.desmos.com/calculator/a9w9srctak
        # self.t = self.p * 0.65 # tooth thickness (mm), very crude self-made estimate (mm)
        # self.l = self.addendum + self.dedendum # tooth length (mm)
        # self.x = self.t**2 / (4 * self.l)

        self.number = number
        if self.number is not None:
            assert self.number in (2, 3, 4, 5)

        # Lewis Form Factor
        # Y_guess = (2 * self.x) / (3 * self.module)
        # Table 14-2, assumes phi = 20 deg, full-depth teeth, and P = 1 teeth/in, m = 25.4 mm/teeth
        # module will be much lower so size effects should be minimal
        # self.Y = np.interp(self.N, Gear.df_14_2["N"], Gear.df_14_2["Y"])

        # self.K_s = 1.192 * (self.F * self.module * np.sqrt(self.Y))**0.0535

        self.C_pf = self.calculate_C_pf()
        self.C_ma = self.calculate_C_ma()
        self.K_H = 1 + Gear.C_mc * (self.C_pf * Gear.C_pm+ self.C_ma * Gear.C_e)

        # Table 14-3, Gear Bending Strength, assumed through or induction hardened, use most conservative function from Fig. 14-2 (grade 1)
        self.S_t = 0.533*self.material.H_B + 88.3 # MPa, N/mm^2

        # Table 14-6, Gear Contact Strength, assumed through or induction hardened, use most conservative function from Fig. 14-5 (grade 1)
        self.S_c = 2.22*self.material.H_B + 200 # MPa, N/mm^2
        # 175000*0.0068947573

        self.Y_theta = 1.0
        # Sec. 14-15
        if self.material.T_C > 120:
            print(f"unknown gear temperature factor b/c {self.material.T_C} > 120 C. {self.Y_theta = } invalid")

        self.Y_Z = Gear.calculate_Y_Z(Gear.reliability)

        # self.sigma_b_all = None
        # self.sigma_c_all = None
        self.sigma_b = None
        self.sigma_c = None

    def K_v(self, V):
        """
        pitch-line velocity V (m/s)
        Eq. 14-27a
        """
        if V > Gear.V_t_max:
            # print(f"exceeding maximum recommended pitch-line velocity: {V = :.4f} > {Gear.V_t_max:.4f} m/s")
            logging.info(f"exceeding maximum recommended pitch-line velocity: {V = :.4f} > {Gear.V_t_max:.4f} m/s")
        return ((Gear.A + np.sqrt(200 * V)) / Gear.A)**Gear.B
    
    def calculate_C_pf(self):
        """Eq. 14-32"""
        F_in = self.F / 25.4
        d_in = self.d / 25.4

        ratio = F_in / (10 * d_in)
        if ratio < 0.05:
            ratio = 0.05

        if F_in <= 1:
            C_pf = ratio - 0.025
        elif F_in <= 17:
            C_pf = ratio - 0.0375 + 0.0125 * F_in
        elif F_in <= 40:
            C_pf = ratio - 0.1109 + 0.0207 * F_in - 0.000228 * F_in**2
        else:
            raise ValueError(f"F = {F_in:.4f} in exceeds the valid range of 40 in")

        return C_pf
    
    def calculate_C_ma(self):
        """Eq. 14-34"""
        F_in = self.F / 25.4
        return Gear.A_14_10 + Gear.B_14_10*F_in + Gear.C_14_10*F_in**2

    @staticmethod
    def calculate_Y_N(n_rpm):
        """Fig 14-14, bending stress-cycle factor"""
        N = L_desired_min * n_rpm # cycles
        return 1.6831*(N**(-0.0323))
    
    @staticmethod
    def calculate_Z_N(n_rpm):
        """Fig 14-15, pitting stress-cycle factor"""
        N = L_desired_min * n_rpm # cycles
        return 2.466*(N**(-0.056))

    @staticmethod
    def calculate_Y_Z(R):
        """
        Eq. 14-38
        reliability (0.5 < R <= 0.9999)
        """
        if 0.5 < R < 0.99:
            Y_Z = 0.658 - 0.0759 * np.log(1 - R)
        elif 0.99 <= R <= 0.9999:
            Y_Z = 0.50 - 0.109 * np.log(1 - R)
        else:
            raise ValueError(f"{R = } is outside the valid range (0.5 < R <= 0.9999)")
        return Y_Z

    def calculate_Z_I(self, m_G):
        """
        Eq 14-23
        m_G (from Eq 14-22)
        """
        phi = np.deg2rad(self.phi)
        # external spur gears assumed
        return (np.cos(phi) * np.sin(phi) / 2) * (m_G / (m_G + 1))

    def bending_stress(self, W_t, V, N_mating):
        """
        Eq. 14-15
        W_t (N), V (m/s), N_mating (teeth)
        returns MPa
        """
        # should be minimum face width between pinion and gear but we set them to be equal
        b = self.F # mm

        # transverse metric module m_t equals regular module for spur gears
        m_t = self.module
        
        # table based on 20 deg pressure angle and full-depth teeth
        # Y_J = FIG_14_6.interp(self.N, N_mating)
        # if np.isnan(Y_J):
        #     Y_J = FIG_14_6.backup_interp(self.N, N_mating)
        Y_J = FIG_14_6.robust_interp([self.N, N_mating])

        return W_t * Gear.K_o * self.K_v(V) * Gear.K_s * (1 / (b * m_t)) * ((self.K_H * Gear.K_B) / Y_J)

    def allowable_bending_stress(self, n_rpm):
        """Eq. 14-17, returns MPa"""
        self.Y_N = Gear.calculate_Y_N(n_rpm)
        return (self.S_t / Gear.S_F) * (self.Y_N / (self.Y_theta * self.Y_Z))
    
    def bending_f_o_s(self, W_t, V, N_mating, n_rpm):
        """
        Eq. 14-41
        W_t (N), V (m/s), N_mating (teeth), n_rpm (rpm)
        """
        self.sigma_b_all = self.allowable_bending_stress(n_rpm)
        self.sigma_b = self.bending_stress(W_t, V, N_mating)
        return self.sigma_b_all / self.sigma_b

    def contact_stress(self, W_t, d_pinion, V, gear_ratio_m):
        """
        Eq. 14-16
        W_t (N), d_pinion (mm), V (m/s), N_mating (teeth)
        returns MPa
        """
        d_w1 = d_pinion # mm

        # should be min face width between pinion and gear but we set them to be equal
        b = self.F # mm

        m_G = gear_ratio_m # Eq 14-22
        Z_I = self.calculate_Z_I(m_G)

        # d_w1 and b must be in mm
        return Gear.Z_E * np.sqrt(
            W_t * Gear.K_o * self.K_v(V) * Gear.K_s * (self.K_H / (d_w1 * b)) * (Gear.Z_R / Z_I)
        )

    def allowable_contact_stress(self, n_rpm):
        """Eq. 14-18, returns MPa"""
        self.Z_N = Gear.calculate_Z_N(n_rpm)
        return (self.S_c / Gear.S_H) * ((self.Z_N * Gear.Z_W) / (self.Y_theta * self.Y_Z))

    def contact_f_o_s(self, W_t, d_pinion, V, gear_ratio_m, n_rpm):
        """
        Eq. 14-42
        W_t (N), d_pinion (mm), V (m/s), N_mating (teeth), n_rpm (rpm)
        """
        self.sigma_c_all = self.allowable_contact_stress(n_rpm)
        self.sigma_c = self.contact_stress(W_t, d_pinion, V, gear_ratio_m)
        return self.sigma_c_all / self.sigma_c

    def print(self):
        pprint.pprint(vars(self))

class GearTrain:
    # k = 1 for full-depth teeth, 0.8 for stub teeth
    k = 1
    def __init__(self, pinion: Gear, gear: Gear, P, n_pinion):
        """P (W), n_pinion (rpm)"""
        self.pinion = pinion
        self.gear = gear

        assert self.pinion.module == self.gear.module
        
        if gear.N % pinion.N != 0:
            print(f"noninteger gear ratio {gear.N = }, {pinion.N = }")
        self.gear_ratio_m = gear.N // pinion.N
        # self.train_value_e = gear_1 / gear_2

        # move this to Gear function?
        self.pinion.n_rpm = n_pinion # rpm
        self.gear.n_rpm = self.pinion.n_rpm * (self.pinion.N / self.gear.N) # rpm
        self.pinion.omega = n_pinion * 2 * np.pi / 60 # rad/s
        self.gear.omega = self.pinion.omega * (self.pinion.N / self.gear.N) # rad/s

        self.pinion.T = P / self.pinion.omega # Nm
        self.gear.T = P / self.gear.omega # Nm

        self.center_d = (self.pinion.d + self.gear.d) / 2 # mm
        pinion_d_m = self.pinion.d / 1e3 # m
        gear_d_m = self.gear.d / 1e3 # m

        self.W_t = self.pinion.T / (pinion_d_m / 2) # N
        assert np.allclose(self.W_t, self.gear.T / (gear_d_m / 2))
        self.W_r = self.W_t * np.tan(np.deg2rad(self.pinion.phi)) # N
        self.W = np.hypot(self.W_t, self.W_r) # N

        self.V = self.pinion.omega * (pinion_d_m / 2) # m/s

        self.pinion.n_b = self.pinion.bending_f_o_s(self.W_t, self.V, self.gear.N, self.pinion.n_rpm)
        self.pinion.n_c = self.pinion.contact_f_o_s(self.W_t, self.pinion.d, self.V, self.gear_ratio_m, self.pinion.n_rpm)
        self.gear.n_b = self.gear.bending_f_o_s(self.W_t, self.V, self.pinion.N, self.gear.n_rpm)
        self.gear.n_c = self.gear.contact_f_o_s(self.W_t, self.pinion.d, self.V, self.gear_ratio_m, self.gear.n_rpm)

    @staticmethod
    def min_teeth(m, phi):
        """
        phi (deg)
        interference
        """
        phi = np.deg2rad(phi)
        return (2*GearTrain.k / ((1 + 2*m)*np.sin(phi)**2)) * (m + np.sqrt(m**2 + (1 + 2*m)*np.sin(phi)**2))
    
    def print(self):
        pprint.pprint(vars(self))

class RetainingRing:
    _loaded = False

    def __init__(self, D):
        """all mm, A-15-16 and A-15-17 for diagrams"""
        RetainingRing.ensure_loaded()
        
        D_idx = (np.abs(RetainingRing.df["D_book"] - D)).argmin()
        row = RetainingRing.df.iloc[D_idx]
        self.D = row["D_book"]

        # assert self.D in ring_df["D_book"]
        # self.D = D
        # row = ring_df[ring_df["D_book"] == self.D]
        
        self.d = row["d_book"].item() # groove diameter
        self.a = row["a_book"].item() # groove thickness
        self.r = row["r_book"].item() # groove fillet radius
        self.t = row["t_book"].item() # groove depth
        # self.t = (self.D - self.d) / 2 # groove depth

        self.a_t = row["a_t_book"].item() # self.a / self.t
        self.r_t = row["r_t_book"].item() # self.r / self.t

        self.K_t  = row["K_t"].item()
        self.K_ts = row["K_ts"].item()

    @staticmethod
    def misumi_rings_df():
        """https://us.misumi-ec.com/vona2/detail/110300258330/"""
        df = pd.read_csv(table_path / "Misumi_Retaining_Rings.csv")
        df["Part Number"] = df["Type"].astype(str) + df["No."].astype(str)
        df["a_book"] = df["m"]
        df["D_book"] = df["d1"]
        df["d_book"] = df["d2"]
        df["t_book"] = (df["D_book"] - df["d_book"]) / 2
        df["r_book"] =  df["t_book"] * 0.2
        return df
    
    @staticmethod
    def mcmaster_rings_df():
        """https://www.mcmaster.com/products/retaining-rings/retaining-rings-2~/external-retaining-rings-7/system-of-measurement~metric/retaining-ring-style~standard/material~1-4122-stainless-steel/"""
        df = pd.read_csv(table_path / "McMaster_Retaining_Rings.csv")
        df["t_book"] = (df["D_book"] - df["d_book"]) / 2
        df["r_book"] =  df["t_book"] * 0.2
        return df

    @staticmethod
    def merge_misumi_mcmaster(misumi_rings, mcmaster_rings):
        df = pd.concat([misumi_rings, mcmaster_rings], join="inner", ignore_index=True)
        df = df[['Part Number', 'd3', 't', 'a_book', 'D_book', 'd_book', 't_book', 'r_book']]
        df["a_t_book"] = df["a_book"] / df["t_book"]
        df["r_t_book"] = df["r_book"] / df["t_book"]
        df["K_t"]  = A_15_16.interp(df[["a_t_book", "r_t_book"]].values)
        df["K_ts"] = A_15_17.interp(df[["a_t_book", "r_t_book"]].values)

        df = df.dropna(ignore_index=True) # drops only if K_t or K_ts is nan

        # Eq 7-3: K_t_over_d^3 is the best indicator of peak stress with just the information of the retaining ring
        df["K_t_over_d^3"] = df["K_t"] / df["d_book"]**3
        # sort by K_t_over_d, drop duplicate D_book values and keep only the first (ensures min K_t_over_d^3 is kept), sort by D_book for further use, clean up indexing

        # display(df.sort_values("K_t").index)
        # display(df.sort_values("K_t_over_d^3").index)
        df = df.sort_values("K_t_over_d^3").drop_duplicates("D_book", keep="first").sort_values("D_book").reset_index(drop=True)
        return df
    
    @classmethod
    def load_data(cls):
        cls.misumi_rings = cls.misumi_rings_df()
        cls.mcmaster_rings = cls.mcmaster_rings_df()
        cls.df = cls.merge_misumi_mcmaster(cls.misumi_rings, cls.mcmaster_rings)
        assert cls.df["D_book"].is_unique # prerequisite for this class to work
        cls._loaded = True

    @classmethod
    def ensure_loaded(cls):
        if not cls._loaded:
            cls.load_data()

    def print(self):
        pprint.pprint(vars(self))

class Key:
    # Table 7-1
    K_t = 2.14
    K_ts = 3.0

    # https://us.misumi-ec.com/vona2/detail/110302681730/?list=PageCategory&seriesCode=110302681730&tab=catalog&Page=1
    # https://us.misumi-ec.com/pdf/fa/2019/2019_US_2442.pdf
    # could go up to 700
    # unknown values filled in with Table A-20

    # old: material = Material("1045 Carbon Steel", 600, 530, 179)
    # 30 HRC -> 277 HB, unknown yield strength so used generic
    material = Material("KESH 1045 Carbon Steel", 700, 530, 277)

    S_sy_MPa = 0.577 * material.S_y_MPa # MPa

    df = pd.read_csv(table_path / "Table_7-6.csv")
    df = df.map(lambda x: float(sum(pd.eval(x.split(" ")))))
    df *= 25.4 # in -> mm

    def __init__(self, d, gear_hub_width: float | None = None, L: float | None = None, w: float | None = None, h: float | None = None):
        """shaft diameter d (mm), gear_hub_width (mm), max length L (mm), width w (mm), height (mm)"""
        assert (gear_hub_width is not None) or (L is not None)
        self.d = d # mm
        self.gear_hub_width = gear_hub_width # None or mm
        self.L = L # mm
        self.w = w # mm
        self.h = h # mm

        self.material = Key.material # this is here so it can be included in table exports

        # If L is provided, it overrides gear_hub_width
        if self.L is None:
            # "Keeping the end of a keyseat at least a distance of d∕10 from the start of the shoulder fillet will prevent the two stress concentrations from combining with each other." - [Shigley, p. 417] cited from [Pilkey, p. 381]
            # subtracting d/10 as to not interfere with shoulder fillet stress concentration
            # subtracting d/10 as to not interfere with retaining ring groove stress concentration
            self.L = self.gear_hub_width - 2*(self.d / 10) # max length, mm
        if self.w is None:
            self.w = max(1.0, np.interp(self.d, Key.df["d"], Key.df["w"])) # mm
        if self.h is None:
            self.h = max(1.0, np.interp(self.d, Key.df["d"], Key.df["h"])) # mm
        
        # length of flat face l
        # should be long enough for shear and crushing fos
        # subtracting w to get length of flat face of rounded key
        self.l = max(1.0, self.L - self.w)

        # self.depth = self.h / 2 # mm

        self.n_shear = None
        self.n_crush = None

        if self.l > self.d * 1.5:
            logging.info("key length over 1.5 * shaft diam.")

    def shear_f_o_s(self, T) -> None:
        """T (N-m)"""
        r = self.d / 2 # mm
        r /= 1000 # m
        F = T / r # N
        A = self.w * self.l # mm^2
        # print(A)
        tau = F / A # N/mm^2, MPa

        # print(f"{r = }, {T = }, {F = }, {A = }, {tau = }")

        self.n_shear = Key.S_sy_MPa / tau
    
    def crush_f_o_s(self, T) -> None:
        """T (N-m)"""
        r = self.d / 2 # mm
        r /= 1000 # m
        F = T / r # N
        A = (self.h / 2) * self.l # mm^2
        sigma = F / A # N/mm^2, MPa

        # print(f"{r = } m, {T = } N-m, {F = } N, {A = } mm^2, {sigma=}")

        self.n_crush = Key.material.S_y_MPa / sigma
    
    def print(self):
        pprint.pprint(vars(self))

class Bearing:
    L_R = 1e6 # Sec 11-2, rated design life, cycles
    a = 3 # Sec 11-3, ball bearing
    reliability = 0.90
    a_1 = 4.26 * np.log(1 / reliability)**(2/3) + 0.05

    # typical r/d for bearing shoulders is 0.02 [p. 402, Shigley]
    r_d_typical = 0.02

    # assumed D/d for optimization purposes
    D_d_assumed = 1.1
    
    def __init__(
            self, 
            width: float, 
            d_shoulder: float, 
            ID: float | None = None, 
            r: float | None = None,
            C_10: float | None = None,
            part_number: str | None = None
        ):
        """width (mm), diameter of shoulder against bearing d_shoulder (mm), bearing ID (mm), bearing corner radius r (mm)"""
        self.width = width # mm
        self.d_shoulder = d_shoulder # mm
        self.ID = ID # mm
        self.r = r # mm
        self.C_10 = C_10
        self.part_number = part_number

        if self.ID is None:
            self.ID = np.floor(self.d_shoulder / Bearing.D_d_assumed)
        if self.r is None:
            self.r = self.d_shoulder * Bearing.r_d_typical

    def calculate_F_R(self, F_D, L_D) -> None:
        """
        desired load F_D (N), desired life L_D (cycles)
        sets dynamic load rating F_R 
        """
        self.F_D = F_D # N
        self.L_D = L_D # cycles
        self.F_R = self.F_D * (self.L_D / (Bearing.a_1 * self.L_R))**(1 / Bearing.a) # N
        if self.C_10 is not None:
            assert self.F_R < self.C_10, "C_10 given as optional argument is greater than calculated rated life"

    def print(self):
        pprint.pprint(vars(self))

class Shaft:
    """
    simply-supported shaft
    bearing A starts at x=0 and bearing B ends at x=L.
    Gear 3 (stage-1 driven gear) at x=L_1
    Gear 4 (stage-2 pinion) at x=L_2
    all forces N, moments N*m, lengths m, diameters mm, pressures MPa when possible
    
    https://excalidraw.com/#json=4ci_CiCLtHcD406ka74OK,dnZGRLAlc2NcMtxoqYGUow
    """
    def __init__(
        self,
        material: Material,
        stage_1: GearTrain,
        stage_2: GearTrain,
        d_shoulder: float,
        L: float,
        L_stage_1,
        L_stage_2,
        bearing_A_width: float,
        bearing_B_width: float,
        extra_bearing_A_kwargs: dict | None = None,
        extra_bearing_B_kwargs: dict | None = None,
        extra_key_1_kwargs: dict | None = None,
        extra_key_2_kwargs: dict | None = None,
    ):
        """d_shoulder (mm), bearing_width (mm), L (m)"""
        self.material = material
        self.stage_1 = stage_1
        self.stage_2 = stage_2

        self.L_cycles = L_desired_min * self.stage_1.gear.n_rpm # cycles

        # self.d_bearing = None
        self.d_gear_1 = self.stage_1.gear.ID
        self.d_gear_2 = self.stage_2.pinion.ID
        self.d_shoulder = d_shoulder

        self.L = L # m
        self.stage_1.L = L_stage_1 # m
        self.stage_2.L = L_stage_2 # m

        if extra_bearing_A_kwargs is None:
            extra_bearing_A_kwargs = {}
        if extra_bearing_B_kwargs is None:
            extra_bearing_B_kwargs = {}

        self.bearing_A = Bearing(bearing_A_width, self.d_gear_1, **extra_bearing_A_kwargs)
        self.bearing_B = Bearing(bearing_B_width, self.d_gear_2, **extra_bearing_B_kwargs)

        self.L_A = self.bearing_A.width / 1e3 / 2 # m 
        self.L_B = self.L - self.bearing_B.width / 1e3 / 2 # m

        self.L_shoulder_1 = self.stage_1.L + (self.stage_1.gear.F / 1e3 / 2) # m
        self.L_shoulder_2 = self.stage_2.L - (self.stage_2.pinion.F / 1e3 / 2) # m

        assert 0 < self.L_A < self.stage_1.L < self.stage_2.L < self.L_B < self.L

        assert np.allclose(self.stage_1.gear.T, self.stage_2.pinion.T)
        self.T = self.stage_1.gear.T

        self.ring_1 = RetainingRing(self.d_gear_1) # stage 1 retaining ring
        self.ring_2 = RetainingRing(self.d_gear_2) # stage 2 retaining ring

        if extra_key_1_kwargs is None:
            extra_key_1_kwargs = {}
        if extra_key_2_kwargs is None:
            extra_key_2_kwargs = {}

        key_1_params = dict(d=self.d_gear_1, gear_hub_width=self.stage_1.gear.F)
        key_1_params.update(extra_key_1_kwargs)
        key_2_params = dict(d=self.d_gear_2, gear_hub_width=self.stage_2.pinion.F)
        key_2_params.update(extra_key_2_kwargs)
        self.key_1 = Key(**key_1_params)
        self.key_2 = Key(**key_2_params)

        try:
            self.L_ring_1 = self.L_shoulder_1 - (self.key_1.gear_hub_width / 1e3)
        except AttributeError as e:
            self.L_ring_1 = self.stage_1.L - (self.stage_1.gear.F / 1e3 / 2)

        try:
            self.L_ring_2 = self.L_shoulder_2 + (self.key_2.gear_hub_width / 1e3)
        except AttributeError as e:
            self.L_ring_2 = self.stage_2.L + (self.stage_2.pinion.F / 1e3 / 2)

        self.R_A_y, self.R_A_z, self.R_B_y, self.R_B_z = Shaft._construct_reactions(self.stage_1.W_r, self.stage_1.W_t, self.stage_2.W_r, self.stage_2.W_t, self.L_A, self.stage_1.L, self.stage_2.L, self.L_B)

        self.R_A = np.hypot(self.R_A_y, self.R_A_z)
        self.R_B = np.hypot(self.R_B_y, self.R_B_z)

        self.bearing_A.calculate_F_R(self.R_A, self.L_cycles)
        self.bearing_B.calculate_F_R(self.R_B, self.L_cycles)

        self.syms, self.numerics = self.construct_functions()

        logging.info("Hardcoded shoulder radius for stage 1 and stage 2 shoulder!")
        self.n_fatigue_shoulder_1, self.n_yield_shoulder_1 = self.shoulder_f_o_s(2.5, self.d_gear_1, self.d_shoulder, self.L_shoulder_1)
        self.n_fatigue_shoulder_2, self.n_yield_shoulder_2 = self.shoulder_f_o_s(2.5, self.d_gear_2, self.d_shoulder, self.L_shoulder_2)

        self.n_fatigue_shoulder_A, self.n_yield_shoulder_A = self.shoulder_f_o_s(self.bearing_A.r, self.bearing_A.ID, self.d_gear_1, self.bearing_A.width / 1e3)
        self.n_fatigue_shoulder_B, self.n_yield_shoulder_B = self.shoulder_f_o_s(self.bearing_B.r, self.bearing_B.ID, self.d_gear_2, self.L - self.bearing_B.width / 1e3)
            
        self.n_fatigue_ring_1, self.n_yield_ring_1 = self.retaining_ring_groove_f_o_s(self.ring_1, self.L_ring_1)
        self.n_fatigue_ring_2, self.n_yield_ring_2 = self.retaining_ring_groove_f_o_s(self.ring_2, self.L_ring_2)

        self.n_fatigue_keyway_1, self.n_yield_keyway_1 = self.keyway_f_o_s(self.key_1, self.stage_1.L)
        self.n_fatigue_keyway_2, self.n_yield_keyway_2 = self.keyway_f_o_s(self.key_2, self.stage_2.L)

        self.key_1.shear_f_o_s(self.T)
        self.key_2.shear_f_o_s(self.T)
        self.key_1.crush_f_o_s(self.T)
        self.key_2.crush_f_o_s(self.T)

    @staticmethod
    def K_fs(K_t, material: Material, r):
        """
        torsion
        K_ts, r (mm)
        """
        return 1 + ((K_t - 1) / (1 + (material.sqrt_a_torsion / np.sqrt(r))))
    
    @staticmethod
    def K_f(K_t, material: Material, r):
        """
        bending / axial
        K_ts, r (mm)
        """
        return 1 + ((K_t - 1) / (1 + (material.sqrt_a_bending / np.sqrt(r))))
    
    @staticmethod
    def _n_fatigue(K_f, K_fs, M_a, M_m, T_a, T_m, d_m, S_e, S_ut):
        """
        DE-Goodman fatigue factor of safety
        d_m (m), S_e (Pa), S_ut (Pa)
        """
        # Eq. 7-6
        A = np.sqrt(4 * (K_f * M_a)**2 + 3 * (K_fs * T_a)**2)
        B = np.sqrt(4 * (K_f * M_m)**2 + 3 * (K_fs * T_m)**2)
        
        # Eq. 7-7
        return (np.pi * d_m**3 / 16) * np.power((A / S_e) + (B / S_ut), -1)
    
    @staticmethod
    def _n_yield(K_f, K_fs, M_a, M_m, T_a, T_m, d_m, S_y):
        """
        DE von Mises yielding factor of safety
        d_m (m), S_y (Pa)
        """
        # Eq. 7-15
        sigma_prime_max = np.sqrt(
            (32 * K_f  * (M_m + M_a) / (np.pi * d_m**3))**2 +
            3 * (16 * K_fs * (T_m + T_a) / (np.pi * d_m**3))**2
        )

        # Eq. 7-16
        return S_y / sigma_prime_max

    @staticmethod
    def _construct_reactions(W_r_1, W_t_1, W_r_2, W_t_2, L_A, L_1, L_2, L_B):
        R_B_y = ((L_2 - L_A) * W_r_2 - (L_1 - L_A) * W_r_1) / (L_B - L_A)
        R_A_y = W_r_2 - W_r_1 - R_B_y

        R_B_z = ((L_1 - L_A) * W_t_1 - (L_2 - L_A) * W_t_2) / (L_B - L_A)
        R_A_z = W_t_1 - W_t_2 - R_B_z

        return R_A_y, R_A_z, R_B_y, R_B_z

    def construct_functions(self):
        _x = sp.Symbol("x", real=True)

        d_discontinuity_A = self.bearing_A.width / 1e3
        d_discontinuity_1 = self.L_shoulder_1
        d_discontinuity_2 = self.L_shoulder_2
        d_discontinuity_B = self.L - self.bearing_B.width / 1e3

        d = sp.Piecewise(
            (self.bearing_A.ID, sp.And(0 <= _x, _x <= d_discontinuity_A)), 
            (self.d_gear_1,     sp.And(d_discontinuity_A < _x, _x <= d_discontinuity_1)), 
            (self.d_shoulder,   sp.And(d_discontinuity_1 < _x, _x < d_discontinuity_2)), 
            (self.d_gear_2,     sp.And(d_discontinuity_2 <= _x, _x < d_discontinuity_B)), 
            (self.bearing_B.ID, sp.And(d_discontinuity_B <= _x, _x <= self.L)), 
            (0, True)
        )
        
        R_Ay, R_By = sp.symbols("R_Ay R_By")
        R_Az, R_Bz = sp.symbols("R_Az R_Bz")
        
        # E and I are placeholders
        E, I = sp.symbols("E I")
        
        beam_y = Beam(self.L, E, I)
        beam_z = Beam(self.L, E, I)

        beam_y.bc_deflection = [(self.L_A, 0), (self.L_B, 0)]
        beam_z.bc_deflection = [(self.L_A, 0), (self.L_B, 0)]

        beam_y.apply_load(R_Ay, self.L_A, -1)
        beam_y.apply_load(self.stage_1.W_r, self.stage_1.L, -1)
        beam_y.apply_load(-self.stage_2.W_r, self.stage_2.L, -1)
        beam_y.apply_load(R_By, self.L_B, -1)

        beam_z.apply_load(R_Az, self.L_A, -1)
        beam_z.apply_load(-self.stage_1.W_t, self.stage_1.L, -1)
        beam_z.apply_load(self.stage_2.W_t, self.stage_2.L, -1) # check again
        beam_z.apply_load(R_Bz, self.L_B, -1)

        beam_y.solve_for_reaction_loads(R_Ay, R_By)
        beam_z.solve_for_reaction_loads(R_Az, R_Bz)

        F_y = beam_y.load
        V_y = beam_y.shear_force()
        M_z = beam_y.bending_moment() # bending moment in y beam is around the z axis

        F_z = beam_z.load
        V_z = beam_z.shear_force()
        M_y = beam_z.bending_moment() # bending moment in z beam is around the y axis

        V = sp.sqrt(V_y**2 + V_z**2)
        M = sp.sqrt(M_y**2 + M_z**2)
        
        T = self.T * sp.SingularityFunction(_x, self.stage_1.L, 0) - self.T * sp.SingularityFunction(_x, self.stage_2.L, 0)
        
        syms = dict(d=d, F_y=F_y, F_z=F_z, V_y=V_y, V_z=V_z, V=V, M_y=M_y, M_z=M_z, M=M, T=T)

        modules = ["numpy"]
        numerics = {
            "d":   sp.lambdify(_x, d,                         modules),
            "F_y": sp.lambdify(_x, F_y.rewrite(sp.Piecewise), modules),
            "F_z": sp.lambdify(_x, F_z.rewrite(sp.Piecewise), modules),
            "V_y": sp.lambdify(_x, V_y.rewrite(sp.Piecewise), modules),
            "V_z": sp.lambdify(_x, V_z.rewrite(sp.Piecewise), modules),
            "V":   sp.lambdify(_x, V.rewrite(sp.Piecewise),   modules),
            "M_y": sp.lambdify(_x, M_y.rewrite(sp.Piecewise), modules),
            "M_z": sp.lambdify(_x, M_z.rewrite(sp.Piecewise), modules),
            "M":   sp.lambdify(_x, M.rewrite(sp.Piecewise),   modules),
            "T":   sp.lambdify(_x, T.rewrite(sp.Piecewise),   modules),
        }

        # beam_y.plot_shear_force()
        # beam_y.plot_bending_moment()
        # beam_z.plot_shear_force()
        # beam_z.plot_bending_moment()
        
        self.reactions = beam_y.reaction_loads | beam_z.reaction_loads
        self.reactions = {k: float(v) for k, v in self.reactions.items()}
        assert np.isclose(self.reactions[R_Ay], self.R_A_y)
        assert np.isclose(self.reactions[R_Az], self.R_A_z)
        assert np.isclose(self.reactions[R_By], self.R_B_y)
        assert np.isclose(self.reactions[R_Bz], self.R_B_z)

        return syms, numerics

    def plot_shaft(self, xs, is_saving=False):
        fig, ax = plt.subplots(dpi=200)

        # shaft
        y = self.numerics["d"](xs) / 2
        ax.fill_between(xs, -y, y, alpha=0.25, color="C0", label="Shaft")
        ax.plot(xs, -y, lw=1.5, color="C0")
        ax.plot(xs, y,  lw=1.5, color="C0")

        # gear 3
        ax.fill_between(
            [self.stage_1.L - (self.stage_1.gear.F / 1e3 / 2), self.L_shoulder_1],
            [-self.stage_1.gear.d / 2, -self.stage_1.gear.d / 2],
            [self.stage_1.gear.d / 2, self.stage_1.gear.d / 2],
            alpha=0.25,
            color="C1",
            label="Gear 3"
        )
        # gear 4
        ax.fill_between(
            [self.L_shoulder_2, self.stage_2.L + (self.stage_2.pinion.F / 1e3 / 2)],
            [-self.stage_2.pinion.d / 2, -self.stage_2.pinion.d / 2],
            [self.stage_2.pinion.d / 2, self.stage_2.pinion.d / 2],
            alpha=0.25,
            color="C2",
            label="Gear 4"
        )

        # ring 1
        ax.fill_between(
            [self.L_ring_1 - self.ring_1.a / 1e3, self.L_ring_1],
            [-self.ring_1.d / 2, -self.ring_1.d / 2],
            [self.ring_1.d / 2, self.ring_1.d / 2],
            alpha=0.25,
            color="C3",
            label="Groove 1"
        )
        # ring 2
        ax.fill_between(
            [self.L_ring_2, self.L_ring_2 + self.ring_2.a / 1e3],
            [-self.ring_2.d / 2, -self.ring_2.d / 2],
            [self.ring_2.d / 2, self.ring_2.d / 2],
            alpha=0.25,
            color="C3",
            label="Groove 2"
        )
        
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$d$ (mm)")
        fig.legend()
        if is_saving:
            fig.savefig(fig_path / "shaft_diameter.png")

    def plot_diagrams(self, show_VMT=False, is_saving=False) -> None:
        xs = np.linspace(0, self.L, 1000)

        self.plot_shaft(xs, is_saving)

        if not show_VMT:
            return None
        
        def label(var: str):
            return f"${var}$ " + ("(N)" if "V" in var else "(N-m)")

        # components
        fig, axs = plt.subplots(4, 1, figsize=(6, 8), sharex=True, dpi=150)
        for i, var in enumerate(("V_y", "V_z", "M_y", "M_z")):
            y = self.numerics[var](xs)

            axs[i].fill_between(xs, y, alpha=0.25)
            axs[i].plot(xs, y, lw=1.5)
            axs[i].grid()
            axs[i].axhline(0, ls='--', color="gray") # lw=0.5
            axs[i].set_ylabel(label(var))
        axs[-1].set_xlabel("$x$ (m)")
        if is_saving:
            fig.savefig(fig_path / "V_M_components.png")

        # resultants
        fig, axs = plt.subplots(3, 1, figsize=(6, 7), sharex=True, dpi=150)
        for i, var in enumerate(("V", "M", "T")):
            y = self.numerics[var](xs)

            axs[i].fill_between(xs, y, alpha=0.25)
            axs[i].plot(xs, y, lw=1.5)
            axs[i].grid()
            axs[i].axhline(0, ls='--', color="gray") # lw=0.5
            axs[i].set_ylabel(label(var))
        axs[-1].set_xlabel("$x$ (m)")
        if is_saving:
            fig.savefig(fig_path / "V_M_T_resultants.png")
        plt.show()

    def shoulder_f_o_s(self, r, d, D, x):
        """r (mm), d (mm), D (mm), x (m)"""

        K_ts = A_15_8.robust_interp((r / d, D / d))
        K_t  = A_15_9.robust_interp((r / d, D / d))
        K_fs = Shaft.K_fs(K_ts, self.material, r)
        K_f = Shaft.K_f(K_t, self.material, r)

        M_a = self.numerics["M"](x) # N-m
        T_a = 0.0
        M_m = 0.0
        T_m = self.numerics["T"](x) # N-m

        return (
            Shaft._n_fatigue(K_f, K_fs, M_a, M_m, T_a, T_m, d / 1e3, self.material.corrected_S_e(d), self.material.S_ut),
            Shaft._n_yield(K_f, K_fs, M_a, M_m, T_a, T_m, d / 1e3, self.material.S_y),
        )
    
    def retaining_ring_groove_f_o_s(self, ring: RetainingRing, x):
        """x (m)"""
        K_fs = Shaft.K_fs(ring.K_ts, self.material, ring.r)
        K_f = Shaft.K_f(ring.K_t, self.material, ring.r)

        M_a = self.numerics["M"](x) # N-m
        T_a = 0.0
        M_m = 0.0
        T_m = self.numerics["T"](x) # N-m

        return (
            Shaft._n_fatigue(K_f, K_fs, M_a, M_m, T_a, T_m, ring.d / 1e3, self.material.corrected_S_e(ring.d), self.material.S_ut),
            Shaft._n_yield(K_f, K_fs, M_a, M_m, T_a, T_m, ring.d / 1e3, self.material.S_y)
        )
    
    def keyway_f_o_s(self, key: Key, x):
        """x (m)"""
        K_fs = Shaft.K_fs(Key.K_ts, self.material, key.d / 2)
        K_f = Shaft.K_f(Key.K_t, self.material, key.d / 2)

        M_a = self.numerics["M"](x) # N-m
        T_a = 0.0
        M_m = 0.0
        T_m = self.T # N-m

        return (
            Shaft._n_fatigue(K_f, K_fs, M_a, M_m, T_a, T_m, key.d / 1e3, self.material.corrected_S_e(key.d), self.material.S_ut),
            Shaft._n_yield(K_f, K_fs, M_a, M_m, T_a, T_m, key.d / 1e3, self.material.S_y)
        )

    def print(self):
        pprint.pprint(vars(self))

# move to Gearbox class
def optim_variables() -> pd.DataFrame:
    df = pd.read_excel("Variables.xlsx")

    mask = df["From List?"]

    df["List"] = df["List"].astype(object)

    df.loc[mask, "List"] = (
        df.loc[mask, "List"]
        .astype(str)
        .apply(lambda x: np.asarray(x.split(",")).astype(np.float64))
    )

    idx = df.index[df["Opt Name"] == "gear_ratio_idx_1"][0]
    df.at[idx, "List"] = df.at[idx, "List"].astype(np.int32)

    df.loc[mask, "Min Bounds"] = 0
    df.loc[mask, "Max Bounds"] = df.loc[mask, "List"].apply(lambda x: len(x)-1)
    df.loc[mask, "Min Bounds"] = df.loc[mask, "Min Bounds"]
    df.loc[mask, "Min Opt Bounds"] = df.loc[mask, "Min Bounds"]
    df.loc[mask, "Max Opt Bounds"] = df.loc[mask, "Max Bounds"]

    df["Opt Bounds Range"] = df["Max Opt Bounds"] - df["Min Opt Bounds"]
    df["Bounds Range"] = df["Max Bounds"] - df["Min Bounds"]
    return df

def export_optim_vars_df(optim_vars_df: pd.DataFrame) -> None:
    export_optim_vars_df = optim_vars_df.loc[
        :,~optim_vars_df.columns.str.contains("Range", case=False)
    ].drop(columns=["Discrete?", "From List?"]) # returns copy

    optim_vars_dfs = {
        "": export_optim_vars_df,
        "_from_list": export_optim_vars_df[optim_vars_df["From List?"]],
        "_discrete": export_optim_vars_df[
            ~optim_vars_df["From List?"] & optim_vars_df["Discrete?"]
        ].drop(columns=["List"]),
        "_continuous": export_optim_vars_df[
            ~optim_vars_df["Discrete?"]
        ].drop(columns=["List"])
    }

    optim_vars_dfs["_from_list"]["List"] = optim_vars_dfs["_from_list"]["List"].apply(lambda x: ", ".join(map(str, x)))

    for end, df in optim_vars_dfs.items():
        df.to_csv(out_table_path / f"opt_vars{end}.csv", index=False)
        df.to_markdown(out_table_path / f"opt_vars{end}.md", index=False)

optim_vars_df = optim_variables()
export_optim_vars_df(optim_vars_df)

def unscale(x, df=optim_vars_df):
    return ((((x - df["Min Opt Bounds"]) / df["Opt Bounds Range"]) * df["Bounds Range"]) + df["Min Bounds"]).to_numpy()

class Gearbox:
    clearance = 5 # mm
    
    def __init__(self, **kwargs):
        try:
            self.df # exists if from_scaled classmethod is used
        except AttributeError as e:
            self.df = pd.DataFrame.from_dict(kwargs, orient="index", columns=["Name", "Value"])
            
        gear_ratio_2 = m_V // kwargs["gear_ratio_1"]

        N_2 = kwargs.get("N_2")
        N_4 = kwargs.get("N_4")
        bearing_A_width = kwargs.get("bearing_A_width")
        bearing_B_width = kwargs.get("bearing_B_width")
        extra_bearing_A_kwargs = kwargs.get("extra_bearing_A_kwargs")
        extra_bearing_B_kwargs = kwargs.get("extra_bearing_B_kwargs")
        extra_key_1_kwargs = kwargs.get("extra_key_1_kwargs")
        extra_key_2_kwargs = kwargs.get("extra_key_2_kwargs")

        if N_2 is None:
            N_2 = np.rint(np.ceil(GearTrain.min_teeth(kwargs["gear_ratio_1"], kwargs["phi_1"]))).astype(int)
        if N_4 is None:
            N_4 = np.rint(np.ceil(GearTrain.min_teeth(gear_ratio_2, kwargs["phi_2"]))).astype(int)

        N_3 = np.rint(kwargs["gear_ratio_1"] * N_2).astype(int)
        N_5 = np.rint(gear_ratio_2 * N_4).astype(int)

        g_2 = Gear(gear_steel, N_2, kwargs["module_1"], kwargs["phi_1"], kwargs["F_1"], number=2)
        g_3 = Gear(gear_steel, N_3, kwargs["module_1"], kwargs["phi_1"], kwargs["F_1"], kwargs["d_shaft_gear_1"], number=3)
        g_4 = Gear(gear_steel, N_4, kwargs["module_2"], kwargs["phi_2"], kwargs["F_2"], kwargs["d_shaft_gear_2"], number=4)
        g_5 = Gear(gear_steel, N_5, kwargs["module_2"], kwargs["phi_2"], kwargs["F_2"], number=5)

        stage_1 = GearTrain(g_2, g_3, P, n_in)
        stage_2 = GearTrain(g_4, g_5, P, stage_1.gear.n_rpm)

        # used as assumptions for optimizer
        if bearing_A_width is None:
            bearing_A_width = 20.0 # mm
        if bearing_B_width is None:
            bearing_B_width = 20.0 # mm

        self.shaft = Shaft(
            shaft_steel, 
            stage_1, 
            stage_2, 
            kwargs["d_shaft_shoulder"], 
            L_intermediate, 
            kwargs["L_1"], 
            kwargs["L_2"],
            bearing_A_width,
            bearing_B_width,
            extra_bearing_A_kwargs,
            extra_bearing_B_kwargs,
            extra_key_1_kwargs,
            extra_key_2_kwargs
        )

        self.depth = (
            Gearbox.clearance
            + self.shaft.stage_1.pinion.addendum
            + self.shaft.stage_1.pinion.d
            + (self.shaft.stage_1.gear.d / 2)
            + (self.shaft.stage_2.pinion.d / 2)
            + self.shaft.stage_2.gear.d
            + self.shaft.stage_2.gear.addendum
            + Gearbox.clearance
        ) # mm

        self.f_o_s()

        self.min_fos_key = min(self.fos_dict, key=self.fos_dict.get)
        self.min_fos = self.fos_dict[self.min_fos_key]

        self.median_fos = np.median(self.fos_arr)

    @staticmethod
    def from_scaled_kwargs(x):
        x = unscale(x)
        # print(x)

        mask_from_list = optim_vars_df["From List?"].to_numpy()
        arr_from_list = optim_vars_df[mask_from_list]["List"].to_numpy()
        idx_from_list = np.rint(x[mask_from_list]).astype(int)

        df = optim_vars_df[["Name", "Opt Name"]].copy()
        df.loc[mask_from_list, "Value"] = np.array([sub[i] for sub, i in zip(arr_from_list, idx_from_list)])

        df.loc[~mask_from_list, "Value"] = x[~mask_from_list] # fill in rest

        # df.set_index("")
        gear_ratio_1 = df[df["Name"] == "gear_ratio_1"]["Value"].item()
        gear_ratio_2 = m_V // gear_ratio_1

        # mask_N_2 = df["Name"].apply(lambda row: row == "N_2")
        mask_N_2 = df["Name"].str.contains("N_2", regex=False)
        df.loc[mask_N_2, "Value"] += np.rint(np.ceil(GearTrain.min_teeth(
            gear_ratio_1, 
            df[df["Name"] == "phi_1"]["Value"].item()
        ))).astype(int)
        mask_N_4 = df["Name"].str.contains("N_4", regex=False)
        df.loc[mask_N_4, "Value"] += np.rint(np.ceil(GearTrain.min_teeth(
            gear_ratio_2, 
            df[df["Name"] == "phi_2"]["Value"].item()
        ))).astype(int)

        if True:
            # stage 1
            # pitch diameter of gear 3 = N_3 * module_1 = gear_ratio_1 * N_2 * module_1
            pitch_d_gear_3 = (
                gear_ratio_1
                * df[df["Name"] == "N_2"]["Value"].item()
                * df[df["Name"] == "module_1"]["Value"].item()
            )
            mask_over_pitch_d_gear_3 = df["Opt Name"].str.contains("over_pitch_d_gear_1", regex=False)
            df.loc[mask_over_pitch_d_gear_3, "Value"] *= pitch_d_gear_3

            # stage 2
            # pitch diameter of gear 4 = N_4 * module_2
            pitch_d_gear_4 = (
                df[df["Name"] == "N_4"]["Value"].item()
                * df[df["Name"] == "module_2"]["Value"].item()
            )
            mask_over_pitch_d_gear_4 = df["Opt Name"].str.contains("over_pitch_d_gear_2", regex=False)
            df.loc[mask_over_pitch_d_gear_4, "Value"] *= pitch_d_gear_4

        mask_over_L = df["Opt Name"].str.contains("over_L", regex=False)
        df.loc[mask_over_L, "Value"] *= L_intermediate

        mask_over_module_1 = df["Opt Name"].str.contains("over_module_1", regex=False)
        df.loc[mask_over_module_1, "Value"] *= df[df["Name"] == "module_1"]["Value"].item()

        mask_over_module_2 = df["Opt Name"].str.contains("over_module_2", regex=False)
        df.loc[mask_over_module_2, "Value"] *= df[df["Name"] == "module_2"]["Value"].item()

        mask_over_max_d_shaft_gear = df["Opt Name"].str.contains("over_max_d_shaft_gear", regex=False)
        max_d_shaft_gear = max(
            df[df["Name"] == "d_shaft_gear_1"]["Value"].item(), 
            df[df["Name"] == "d_shaft_gear_2"]["Value"].item()
        )
        df.loc[mask_over_max_d_shaft_gear, "Value"] *= max_d_shaft_gear

        kwargs = dict(zip(df["Name"], df["Value"]))

        # print(df)
        # print(kwargs)
        return df, kwargs

    @classmethod
    def from_scaled(cls, x):
        """
        construct Gearbox object from scaled/normalized input vector x
        used with optimization variables described in Variables.xlsx
        """
        cls.df, cls.kwargs = cls.from_scaled_kwargs(x)
        return cls(**cls.kwargs) # calls the __init__ constructor
    
    def f_o_s(self):
        """creates fos_dict and fos_arr"""
        self.fos_dict = {
            "stage_1.pinion.n_b":   self.shaft.stage_1.pinion.n_b, # could remove
            "stage_1.pinion.n_c":   self.shaft.stage_1.pinion.n_c, # could remove
            "stage_1.gear.n_b":     self.shaft.stage_1.gear.n_b,
            "stage_1.gear.n_c":     self.shaft.stage_1.gear.n_c,
            "stage_2.pinion.n_b":   self.shaft.stage_2.pinion.n_b,
            "stage_2.pinion.n_c":   self.shaft.stage_2.pinion.n_c,
            "stage_2.gear.n_b":     self.shaft.stage_2.gear.n_b, # could remove
            "stage_2.gear.n_c":     self.shaft.stage_2.gear.n_c, # could remove
            "n_fatigue_ring_1":     self.shaft.n_fatigue_ring_1,
            "n_fatigue_ring_2":     self.shaft.n_fatigue_ring_2,
            "n_fatigue_shoulder_1": self.shaft.n_fatigue_shoulder_1,
            "n_fatigue_shoulder_2": self.shaft.n_fatigue_shoulder_2,
            "n_fatigue_shoulder_A": self.shaft.n_fatigue_shoulder_A,
            "n_fatigue_shoulder_B": self.shaft.n_fatigue_shoulder_B,
            "n_fatigue_keyway_1":   self.shaft.n_fatigue_keyway_1,
            "n_fatigue_keyway_2":   self.shaft.n_fatigue_keyway_2,
            "n_yield_ring_1":       self.shaft.n_yield_ring_1,
            "n_yield_ring_2":       self.shaft.n_yield_ring_2,
            "n_yield_shoulder_1":   self.shaft.n_yield_shoulder_1,
            "n_yield_shoulder_2":   self.shaft.n_yield_shoulder_2,
            "n_yield_shoulder_A":   self.shaft.n_yield_shoulder_A,
            "n_yield_shoulder_B":   self.shaft.n_yield_shoulder_B,
            "n_yield_keyway_1":     self.shaft.n_yield_keyway_1,
            "n_yield_keyway_2":     self.shaft.n_yield_keyway_2,
            "key_1.n_shear":        self.shaft.key_1.n_shear,
            "key_2.n_shear":        self.shaft.key_2.n_shear,
            "key_1.n_crush":        self.shaft.key_1.n_crush,
            "key_2.n_crush":        self.shaft.key_2.n_crush,
        }

        self.fos_arr = np.array(list(self.fos_dict.values()))

    def export_input_parameters(self, file_name: str) -> None:
        df = self.df.copy()
        df = df[["Name", "Value"]]

        df["Value"] = df["Value"].apply(format_value)

        export_latex_table(df[["Name", "Value"]], file_name) 

    def export_CAD_parameters(self):
        data = dict(
            L                 = self.shaft.L   * 1e3,
            L_A_end           = self.shaft.bearing_A.width,
            L_ring_1          = self.shaft.L_ring_1 * 1e3,
            L_shoulder_1      = self.shaft.L_shoulder_1 * 1e3,
            L_shoulder_2      = self.shaft.L_shoulder_2 * 1e3,
            L_ring_2          = self.shaft.L_ring_2 * 1e3,
            L_B_end           = self.shaft.L * 1e3 - self.shaft.bearing_B.width,
            r_gear_1          = self.shaft.d_gear_1 / 2,
            r_shoulder        = self.shaft.d_shoulder / 2,
            r_gear_2          = self.shaft.d_gear_2 / 2,
            r_ring_1          = self.shaft.ring_1.d / 2,
            a_ring_1          = self.shaft.ring_1.a,
            radius_ring_1     = self.shaft.ring_1.r,
            r_ring_2          = self.shaft.ring_2.d / 2,
            a_ring_2          = self.shaft.ring_2.a,
            radius_ring_2     = self.shaft.ring_2.r,
            r_A               = self.shaft.bearing_A.ID / 2,
            r_B               = self.shaft.bearing_B.ID / 2,
            radius_shoulder_A = self.shaft.bearing_A.r,
            radius_shoulder_B = self.shaft.bearing_B.r
        )
        df = pd.DataFrame.from_dict(data, orient="index").reset_index()
        df.columns = ["Parameter", "Value"]
        df.to_excel("Parameters.xlsx", index=False)

        print("Exported Parameters")

    def export_tables(self):
        df_gearbox = get_instance_variables_df(self)
        
        df_shaft = get_instance_variables_df(self.shaft)
        
        df_stage_1 = get_instance_variables_df(self.shaft.stage_1)
        df_stage_2 = get_instance_variables_df(self.shaft.stage_2)
        df_stages = pd.merge(df_stage_1, df_stage_2, on=["Variable", "Unit"], suffixes=(" Stage 1", " Stage 2"))

        gear_dfs = [
            get_instance_variables_df(self.shaft.stage_1.pinion).set_index(["Variable", "Unit"]),
            get_instance_variables_df(self.shaft.stage_1.gear).set_index(["Variable", "Unit"]),
            get_instance_variables_df(self.shaft.stage_2.pinion).set_index(["Variable", "Unit"]),
            get_instance_variables_df(self.shaft.stage_2.gear).set_index(["Variable", "Unit"])
        ]
        df_gears = pd.concat(gear_dfs, axis=1).reset_index()
        df_gears.columns = ["Variable", "Unit", "Gear 2 (P1)", "Gear 3 (G1)", "Gear 4 (P2)", "Gear 5 (G2)"]

        df_r1 = get_instance_variables_df(self.shaft.ring_1)
        df_r2 = get_instance_variables_df(self.shaft.ring_2)
        df_rings = pd.merge(df_r1, df_r2, on=["Variable", "Unit"], suffixes=(" Ring 1", " Ring 2"))

        df_k1 = get_instance_variables_df(self.shaft.key_1)
        df_k2 = get_instance_variables_df(self.shaft.key_2)
        df_keys = pd.merge(df_k1, df_k2, on=["Variable", "Unit"], suffixes=(" Key 1", " Key 2"))

        df_bA = get_instance_variables_df(self.shaft.bearing_A)
        df_bB = get_instance_variables_df(self.shaft.bearing_B)
        df_bearings = pd.merge(df_bA, df_bB, on=["Variable", "Unit"], suffixes=(" Bearing A", " Bearing B"))

        tables = {
            "gearbox": df_gearbox,
            "shaft": df_shaft,
            "stages": df_stages,
            "gears": df_gears,
            "rings": df_rings,
            "keys": df_keys,
            "bearings": df_bearings
        }

        order = var_unit_map_df["LaTeX_Var"].unique().tolist()

        for name, df in tables.items():
            val_cols = [c for c in df.columns if c not in ["Variable", "Unit"]]
            df = df[["Variable", "Unit"] + val_cols]

            # sort rows based on CSV order
            # use Categorical to force the sort order to match the "order" list
            df["Variable"] = pd.Categorical(df["Variable"], categories=order, ordered=True)
            df = df.sort_values("Variable")

            for col in df.columns:
                if col not in ["Variable", "Unit"]:
                    df[col] = df[col].apply(format_value)
            
            export_latex_table(df, name)
            print(f"Exported {name}.tex")