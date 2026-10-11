function b6_matlab(outCsv, sceneJson)
%B6_MATLAB  B6 cross-tool timing: one RIS channel evaluation in MATLAB (DESIGN.md 12.2).
%   b6_matlab(OUTCSV, SCENEJSON) reads the B6 scene (written by bench/b6_crosstool.py),
%   times, for every panel size N in the scene, one channel evaluation h (RIS sum + LOS)
%   for the same Tx, Rx, carrier and 1-bit configuration:
%     hand_vec   vectorized per-element K1 sum (hand formula, no toolbox calls)
%     hand_loop  the same sum as an explicit per-element for-loop (a "per-element script")
%     toolbox    golden_util_k1 (rangeangle, phased.CosineAntennaElement, fspl; includes an
%                independent hand evaluation and the LOS term), sremu tools/matlab
%     risreflector  one phased.RISReflector step (far-field plane wave, unit-gain cells)
%   with timeit (median of repeated calls), and writes one CSV row per (N, method) with
%   |h| in dB of the K1 paths (RIS sum + LOS) as a same-scene check against sremu.
%   Scope: RISReflector is plane-wave (far field) and has no cell-area factor; it evaluates a
%   different model and is timed only as tool context.
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'tools', 'matlab'));
sc = jsondecode(fileread(sceneJson));
c0 = physconst('LightSpeed');
fc = sc.fc_hz;
lam = c0 / fc;
tx = sc.tx_m(:);
rx = sc.rx_m(:);
iso.alpha = NaN; iso.boresight = [1; 0; 0];
rows = {};
for i = 1:numel(sc.panels)
    pn = sc.panels(i);
    P = golden_util_panel([0 0 0], [1 0 0], [0 1 0], pn.n_u, pn.n_v, pn.pitch_m, pn.pitch_m, 1, 1, 0);
    gam = reshape(double(pn.gamma(:)), 1, []);
    N = pn.n_u * pn.n_v;
    f1 = @() hand_vec(tx, rx, P, fc, gam);
    f2 = @() hand_loop(tx, rx, P, fc, gam);
    f3 = @() golden_util_k1(tx, rx, P, fc, gam, iso, iso);
    h1 = f1(); h2 = f2(); [h3r, ~, h3los] = f3(); h3 = h3r + h3los;
    rows(end+1, :) = {N, 'matlab_hand_vec', timeit(f1), 20*log10(abs(h1))}; %#ok<AGROW>
    rows(end+1, :) = {N, 'matlab_hand_loop', timeit(f2), 20*log10(abs(h2))}; %#ok<AGROW>
    rows(end+1, :) = {N, 'matlab_toolbox_golden_util_k1', timeit(f3), 20*log10(abs(h3))}; %#ok<AGROW>
    % far-field RISReflector: URA in the y-z plane, boresight +x (same orientation)
    ris = phased.RectangularRIS('Size', [pn.n_v pn.n_u], 'Spacing', [pn.pitch_m pn.pitch_m], ...
        'UnitCell', phased.IsotropicAntennaElement);
    refl = phased.RISReflector('Surface', ris, 'OperatingFrequency', fc, 'PropagationSpeed', c0);
    [~, angIn] = rangeangle(tx, [0; 0; 0]);
    [~, angOut] = rangeangle(rx, [0; 0; 0]);
    w = gam(:);
    f4 = @() refl(1, angIn, angOut, w);
    f4(); % lock the System object (setup) before timing the step
    rows(end+1, :) = {N, 'matlab_risreflector_step', timeit(f4), NaN}; %#ok<AGROW>
    fprintf('N=%d  hand_vec %.3g s  hand_loop %.3g s  toolbox %.3g s  RISReflector %.3g s\n', ...
        N, rows{end-3, 3}, rows{end-2, 3}, rows{end-1, 3}, rows{end, 3});
end
T = cell2table(rows, 'VariableNames', {'N', 'method', 't_median_s', 'h_abs_db'});
T.matlab_release = repmat({version('-release')}, height(T), 1);
writetable(T, outCsv);
end

function h = hand_vec(tx, rx, P, fc, gam)
c0 = physconst('LightSpeed');
k = 2 * pi * fc / c0;
vi = tx - P.pos; vs = rx - P.pos;
ri = vecnorm(vi); rs = vecnorm(vs);
ci = (P.normal.' * vi) ./ ri;
cs = (P.normal.' * vs) ./ rs;
o = sqrt(max(ci, 0) .* max(cs, 0));
t = gam .* P.area .* o ./ (4 * pi * ri .* rs) .* 1j .* exp(-1j * k * (ri + rs));
d = norm(rx - tx);
h = sum(t) + c0 / fc / (4 * pi * d) * exp(-1j * k * d);
end

function h = hand_loop(tx, rx, P, fc, gam)
c0 = physconst('LightSpeed');
k = 2 * pi * fc / c0;
h = 0;
n = P.normal;
for q = 1:size(P.pos, 2)
    p = P.pos(:, q);
    vi = tx - p; vs = rx - p;
    ri = norm(vi); rs = norm(vs);
    ci = (n.' * vi) / ri; cs = (n.' * vs) / rs;
    if ci > 0 && cs > 0
        h = h + gam(q) * P.area(q) * sqrt(ci * cs) / (4 * pi * ri * rs) * 1j * exp(-1j * k * (ri + rs));
    end
end
d = norm(rx - tx);
h = h + c0 / fc / (4 * pi * d) * exp(-1j * k * d);
end
