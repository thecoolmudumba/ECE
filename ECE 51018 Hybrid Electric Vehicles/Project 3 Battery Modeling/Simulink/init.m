clearvars;
load us06.txt

% Cell capacity
Capacity = 31; % Ampere-hours
Q_cell = Capacity*3600; % cell capacity in Coulombs

% R0 resistance 
R0 = 0.009;

% R1 Resistance
R1 = 0.0015; %Ohms

% C1 Capacitance 
C1 = 3.5e4; %Farads

Ns = 100;  % Number of series cells
Np = 6;   % Number of parallel cells

load P_trac.mat
SOC_min = 0.2;