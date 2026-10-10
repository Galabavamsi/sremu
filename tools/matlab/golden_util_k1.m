function [h, hHand, hLos, chk] = golden_util_k1(tx, rx, P, fc, gam, txAnt, rxAnt)
%GOLDEN_UTIL_K1  K1 RIS coefficient (DESIGN.md section 5.3, cell_model "point") built from
%   toolbox primitives, plus an independent hand-formula evaluation and the LOS coefficient.
%
%   h = sum_n sqrt(G_T G_R) Gamma_n A_n o(theta_i,theta_s)/(4 pi r_i r_s) j exp(-jk(r_i+r_s))
%   over visible cells (cos theta_i > 0 and cos theta_s > 0), o = sqrt(cos_i cos_s).
%
%   Toolbox path: rangeangle in the panel axes (local x = normal) gives r and the incidence
%   and scattering directions; phased.CosineAntennaElement with CosinePower [0.5 0.5] gives
%   the voltage (field) pattern sqrt(cos) per direction; fspl gives lambda/(4 pi r) per hop and
%   the cell gain 4 pi A/lambda^2 turns the two Friis hops into A/(4 pi r_i r_s).
%   Antennas: struct alpha (NaN = isotropic) and boresight (3x1); CosPower(alpha) is
%   phased.CosineAntennaElement with CosinePower [alpha/2 alpha/2] scaled by G0 = 2(alpha+1).
%   All complex outputs include the propagation phase exp(-jk path).
c0 = physconst('LightSpeed');
lam = c0 / fc;
k = 2 * pi / lam;
gam = reshape(gam, 1, []);
cellEl = phased.CosineAntennaElement('CosinePower', [0.5 0.5]);

[ri, angI] = rangeangle(tx(:), P.pos, P.axes);      % Tx seen from each cell (panel axes)
[rs, angS] = rangeangle(rx(:), P.pos, P.axes);      % Rx seen from each cell
oi = reshape(cellEl(fc, angI), 1, []);
os = reshape(cellEl(fc, angS), 1, []);
vi = tx(:) - P.pos; vs = rx(:) - P.pos;
ci = (P.normal.' * vi) ./ vecnorm(vi);
cs = (P.normal.' * vs) ./ vecnorm(vs);
vis = ci > 0 & cs > 0;
gT = antenna_power_gain(txAnt, fc, tx(:), P.pos);
gR = antenna_power_gain(rxAnt, fc, rx(:), P.pos);
Li = db2mag(-fspl(ri, lam));
Ls = db2mag(-fspl(rs, lam));
gCell = 4 * pi * P.area / lam^2;
terms = sqrt(gT .* gR) .* gam .* gCell .* oi .* os .* Li .* Ls .* 1j .* exp(-1j * k * (ri + rs));
terms(~vis) = 0;
h = sum(terms);

% independent hand formula (no toolbox calls)
rih = vecnorm(vi); rsh = vecnorm(vs);
o = sqrt(max(ci, 0) .* max(cs, 0));
th = sqrt(gT .* gR) .* gam .* P.area .* o ./ (4 * pi * rih .* rsh) .* 1j .* exp(-1j * k * (rih + rsh));
th(~vis) = 0;
hHand = sum(th);

% LOS (section 5.2): sqrt(G_T(r^) G_R(-r^)) lambda/(4 pi d) exp(-jkd)
d = rangeangle(rx(:), tx(:));
gTl = antenna_power_gain(txAnt, fc, tx(:), rx(:));
gRl = antenna_power_gain(rxAnt, fc, rx(:), tx(:));
hLos = sqrt(gTl * gRl) * db2mag(-fspl(d, lam)) * exp(-1j * k * d);

chk.cosine_element_max_dev = max(abs([oi(vis).^2 - ci(vis), os(vis).^2 - cs(vis)]));
chk.range_max_dev_m = max(abs([ri - rih, rs - rsh]));
end

function g = antenna_power_gain(ant, fc, from, to)
% Power gain of the antenna at FROM toward each column of TO.
m = size(to, 2);
if isnan(ant.alpha)
    g = ones(1, m);
    return
end
b = ant.boresight(:) / norm(ant.boresight);
a = [0; 0; 1];
if abs(dot(a, b)) > 0.9, a = [0; 1; 0]; end
y = cross(a, b); y = y / norm(y);
z = cross(b, y);
[~, ang] = rangeangle(to, from, [b, y, z]);
el = phased.CosineAntennaElement('CosinePower', [ant.alpha / 2, ant.alpha / 2]);
r = reshape(el(fc, ang), 1, []);
g = 2 * (ant.alpha + 1) * abs(r).^2;
end
