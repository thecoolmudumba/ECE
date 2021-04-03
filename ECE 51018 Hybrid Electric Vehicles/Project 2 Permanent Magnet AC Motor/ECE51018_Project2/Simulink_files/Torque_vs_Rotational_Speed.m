clear all

% motor parameters
P = 6; % number of poles
lambda_m = 0.1062;  %flux constant V-s/rad
r_s = 0.01;  % stator resistance in ohms
L_d = 0.3e-3; %d-axis inductance in H
L_q = 0.3e-3; %q-axis inductance in H

% Filter parameters
L = 5e-6; % inductance in H
R = 0.01; % resistance in ohms
C = 1e-3; % capacitance in F

V_batt = 350;  % battery voltage

% define electrical rotor speed
omega_r = 3000*2*pi/60 * P/2; % rad/s


% set q-axis current
I_qs_star = 0;  %replace

%set d-axis current to zero
I_ds_star = 0;


% Current Control gains
Kq = 0.5;
Kd = 0.5;

N_w = 100;
N_i = 1000;
I_max = 250; % maximum current in A
V_max = V_batt/sqrt(3); % maximum voltage in V

w_r = linspace(0, 1500, N_w); % electrical rotor speed in radians per second
I_ds = -linspace(I_max, 0, N_i); % in Amps

for i = 1:N_w
    Te_max = 0;
    % perform one-dimensional search for optimal i_ds, i_qs
    for j = 1:N_i
        I_qs = sqrt(I_max^2 - I_ds(j)^2);
        V_qs = r_s*I_qs + w_r(i)*(L_d*I_ds(j) + lambda_m);
        V_ds = r_s*I_ds(j) - w_r(i)*L_q*I_qs;
        
        V_p = sqrt(V_qs^2 + V_ds^2);
        
        if (V_p < V_max) % viable but not necessarily optimal solution
            Te = 1.5*(P/2)*(lambda_m*I_qs + (L_d-L_q)*I_qs*I_ds(j));
            if (Te >= Te_max) % best viable solution thus far
                Te_max = Te;
                w_r(i);
                % save optimal I_ds for plotting vs speed
                optimum_I_ds(i) = I_ds(j);
            end
        end
        V_p;
    end
    Te(i) = Te_max;  % save for plotting vs speed
end

w_rm = (2/P)*w_r;
figure(1)
plot(w_rm,Te) 
