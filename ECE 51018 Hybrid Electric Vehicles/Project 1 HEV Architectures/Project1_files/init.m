clear
M_glider = 1746;  % glider mass, kg
M_driver = 70; % driver mass kg
C_D = 0.35;    % drag coefficient
C_0 = 0.015;   % rolling resistance coefficient
A_F = 1.93;    % frontal area, m^2
eta_ess = 0.8; % energy storage subsystem round-trip efficiency
r_wheel = 0.2794; % wheel radius, m
P_eng_min = 10000.0; % min engine power, W
P_eng_max = 85000.0; % max engine power, W
G_diff = 0.268;     % differential gear ratio
G_cvt_min = 0.5;  % minimum tansmission gear ratio
w_eng_min = 1000; % minimum engine speed, rpm
E_batt = 2.0;   % battery capacity, kW
SOC_init = 0.5; % initial SOC
m_batt = 21.74; % battery mass, kg/kw-hr
G_elec = 1; % gear ratio for motor

% physical constants
rho = 1.225; % density of air, kg/m^3
g = 9.8; % acceleration due to gravity,  m/s^s

% unit conversions
m_to_mi = 1/1609; % meters to miles
g_per_hr_to_gal_per_s = 9.778e-8;  % g/hr to gal/s
mi_per_hr_to_m_per_s = 0.44704; % mi/hr to m/s

% calculated constants

v_veh_min = G_cvt_min * G_diff*r_wheel * w_eng_min * 2 * pi / 60;  % smallest vehicle speed for engine to stay engaged, in m/s
E_ess_max = E_batt * 1000 * 3600; % energy storage capacity, J
M_batt = m_batt* E_batt;
M_veh = M_glider + M_driver + M_batt;

% load drive cycles
load hwfet.txt
load ftp75.txt
load us06.txt
load eng_map.txt
load la92.txt

% wrap parameters into structure
param.v_veh_min = v_veh_min ;
param.P_eng_min = P_eng_min ;
param.P_eng_max = P_eng_max ; 
param.G_cvt_min = G_cvt_min ;
param.G_diff = G_diff ;    
param.r_wheel = r_wheel ;