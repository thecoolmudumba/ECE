figure(1)
x = Vbatt.time;
y = Vbatt.signals.values;
plot(x,y)
grid on
title('V_{batt} versus time')
ylabel('V_{batt} (V)')
xlabel('time (s)')

figure(2)
x = Pbatt.time;
y = Pbatt.signals.values/1000;
plot(x,y)
grid on
title('P_{batt} versus time')
ylabel('P_{batt} (kW)')
xlabel('time (s)')

figure(3)
x = Ibatt.time;
y = Ibatt.signals.values;
plot(x,y)
grid on
title('I_{batt} versus time')
ylabel('I_{batt} (A)')
xlabel('time (s)')

figure(4)
x = SOC.time;
y = SOC.signals.values;
plot(x,y)
grid on
title('SOC versus time')
ylabel('SOC')
xlabel('time (s)')

figure(5)
x = Ebatt.time;
y = Ebatt.signals.values/1000/3600;
plot(x,y)
grid on
title('E_{batt} versus time')
ylabel('E_{batt} (kWh)')
xlabel('time (s)')
