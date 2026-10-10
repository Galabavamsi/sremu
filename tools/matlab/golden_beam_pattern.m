function golden_beam_pattern(outDir)
%GOLDEN_BEAM_PATTERN  G2 and G3 goldens: phase-gradient RIS beam pattern (DESIGN.md 10.6).
%   16 x 16 cells at lambda/2, 3.5 GHz, incidence from az = -30 deg, steered to +20 deg, Tx and
%   Rx at R = 500 m in the azimuth plane, outgoing az = -85:0.25:85 deg.
%   Code paths: (P2) phased.RectangularRIS + phased.RISReflector with a
%   phased.CosineAntennaElement CosinePower [0.5 0.5] unit cell (so the element factor is
%   sqrt(cos theta_i cos theta_s), as in K1); (P3) the hand array factor
%   sum_n w_n E(in) E(out) exp(jk p_n.(u_in + u_out)); (K1) the exact near-field K1 sum at
%   500 m. Profiles: G2 "continuous" phases quantized to 4 bits (16 states, so the emulator
%   can express them with an ideal 4-bit table) and G3 the 1-bit {0, pi} profile.
%   MATLAB's RIS cells have unit gain: |h_K1| = |h_P2 channel| * 4 pi dx dy / lambda^2.
%   Writes G2_beam_pattern.csv, G3_beam_pattern_1bit.csv, G2_G3_cells.csv and meta files.
if nargin < 1, outDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'golden', 'matlab_R2026a'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end
tStart = tic;
c0 = physconst('LightSpeed');
fc = 3.5e9; lam = c0 / fc; k = 2 * pi / lam;
N = 16; d = lam / 2; R = 500;
azIn = -30; azTarget = 20; az = -85:0.25:85;
nBits = 4;

cellEl = phased.CosineAntennaElement('CosinePower', [0.5 0.5]);
ris = phased.RectangularRIS('Size', [N N], 'Spacing', [d d], 'UnitCell', cellEl);
ura = phased.URA('Size', [N N], 'ElementSpacing', [d d]);
p = getElementPosition(ura);                 % 3 x N^2, y-z plane, centred, boresight +x
% map URA elements to the sremu cell order n = iv*N + iu (u = +y, v = n x u = +z)
iu = round(p(2, :) / d + (N - 1) / 2);
iv = round(p(3, :) / d + (N - 1) / 2);
nIdx = iv * N + iu;
assert(isequal(sort(nIdx), 0:N^2-1), 'golden:g2', 'URA -> cell mapping is not a permutation');
P = golden_util_panel([0 0 0], [1 0 0], [0 1 0], N, N, d, d, 1, 1, 0);
assert(max(abs(P.pos(:, nIdx + 1) - p), [], 'all') < 1e-15, 'golden:g2', 'panel geometry mismatch');

uIn = [cosd(azIn); sind(azIn); 0];
uT = [cosd(azTarget); sind(azTarget); 0];
wIdeal = exp(-1j * k * (p.' * (uIn + uT)));     % co-phase toward the target
sCont = mod(round(angle(wIdeal) / (2 * pi / 2^nBits)), 2^nBits);
wCont = exp(1j * 2 * pi * sCont / 2^nBits);
phiq = mod(angle(wIdeal), 2 * pi);
sBit = double(phiq >= pi / 2 & phiq < 3 * pi / 2);
wBit = exp(1j * pi * sBit);

tx = R * uIn;
refl = phased.RISReflector('Surface', ris, 'OperatingFrequency', fc, 'PropagationSpeed', c0);
iso = struct('alpha', NaN, 'boresight', [1; 0; 0]);
profiles = {'G2_beam_pattern', 'continuous_4bit', wCont, sCont, 0:2^nBits-1; ...
            'G3_beam_pattern_1bit', 'onebit', wBit, sBit, [0 1]};
cellOut = golden_util_write_csv(fullfile(outDir, 'G2_G3_cells.csv'), ...
    {'n', 'iu', 'iv', 'x', 'y', 'z', 'state_continuous_4bit', 'state_onebit'}, ...
    {(0:N^2-1).', P.iu.', P.iv.', P.pos(1, :).', P.pos(2, :).', P.pos(3, :).', ...
     reorder(sCont, nIdx).', reorder(sBit, nIdx).'});

for Q = 1:size(profiles, 1)
    [gid, pname, w, s, ~] = profiles{Q, :};
    gam = reorder(w.', nIdx);                    % Gamma per sremu cell
    yRefl = zeros(size(az)); yHand = yRefl; hK1 = yRefl; rxM = zeros(3, numel(az));
    Ein = cellEl(fc, [azIn; 0]);
    for a = 1:numel(az)
        uOut = [cosd(az(a)); sind(az(a)); 0];
        yRefl(a) = refl(1, [azIn; 0], [az(a); 0], w);
        release(refl);
        yHand(a) = Ein * cellEl(fc, [az(a); 0]) * sum(w .* exp(1j * k * (p.' * (uIn + uOut))));
        rx = R * uOut; rxM(:, a) = rx;
        hK1(a) = golden_util_k1(tx, rx, P, fc, gam, iso, iso);
    end
    % unit-gain-cell channel of the reflector path: y (lambda/4pi)^2/(Rt Rr) exp(-jk(Rt+Rr))
    hRefl = yRefl * (lam / (4 * pi))^2 / (R * R) * exp(-1j * k * 2 * R);
    lvl = 20 * log10(abs(yRefl));
    [mainAz, mainDb] = parabolic_peak(az, lvl, az > 0 & az < 35);
    out = golden_util_write_csv(fullfile(outDir, [gid '.csv']), ...
        {'az_deg', 'rx_x', 'rx_y', 'rx_z', 'y_reflector_re', 'y_reflector_im', 'y_hand_af_re', 'y_hand_af_im', ...
         'h_reflector_channel_re', 'h_reflector_channel_im', 'h_k1_re', 'h_k1_im'}, ...
        {az.', rxM(1, :).', rxM(2, :).', rxM(3, :).', real(yRefl).', imag(yRefl).', real(yHand).', imag(yHand).', ...
         real(hRefl).', imag(hRefl).', real(hK1).', imag(hK1).'});
    checks = struct('reflector_vs_hand_af_max_rel', max(abs(yRefl - yHand)) / max(abs(yHand)), ...
        'main_lobe_az_deg', mainAz, 'main_lobe_db', mainDb, ...
        'k1_minus_reflector_channel_db_at_main_lobe', 20 * log10(abs(hK1(az == 20)) / abs(hRefl(az == 20))), ...
        'cell_gain_constant_db', 20 * log10(4 * pi * d * d / lam^2));
    if strcmp(pname, 'onebit')
        [mirAz, mirDb] = parabolic_peak(az, lvl, az > 30 & az < 60);
        checks.mirror_lobe_az_deg = mirAz;
        checks.mirror_lobe_db_vs_main = mirDb - mainDb;
    end
    params = struct('fc_hz', fc, 'n_u', N, 'n_v', N, 'pitch_m', d, 'range_m', R, 'az_in_deg', azIn, ...
        'az_target_deg', azTarget, 'az_out_deg', [az(1) az(end) 0.25], 'profile', pname, ...
        'n_states', numel(profiles{Q, 5}), 'alphabet', ifelse(strcmp(pname, 'onebit'), 'ideal_antipodal {+1, -1}', ...
        sprintf('ideal %d-bit exp(j 2 pi s/%d)', nBits, 2^nBits)), ...
        'panel', struct('center_m', [0 0 0], 'normal', [1 0 0], 'u_axis', [0 1 0], 'cell_model', 'point'), ...
        'tx_m', tx.', 'unit_cell', 'phased.CosineAntennaElement CosinePower [0.5 0.5]', ...
        'weights_rule', 'w = exp(-jk p.(u_in + u_target)) quantized; 1-bit: pi where mod(angle(w),2pi) in [pi/2, 3pi/2)');
    notes = {'G2/G3 show code agreement only: MATLAB unit-gain cells share v2''s normalization, so they do not validate absolute RIS power (DESIGN.md 10.6).', ...
        'Peaks by parabolic interpolation of the dB pattern of the reflector path (y_reflector).', ...
        'States per cell are in G2_G3_cells.csv (cell order n = iv*16 + iu).'};
    golden_util_write_meta(outDir, gid, mfilename, toc(tStart), params, checks, {out, cellOut}, notes);
    fprintf('%s: main lobe %.3f deg; reflector vs hand %.2e\n', gid, mainAz, checks.reflector_vs_hand_af_max_rel);
    if strcmp(pname, 'onebit')
        fprintf('   mirror lobe %.3f deg, %.3f dB vs main\n', checks.mirror_lobe_az_deg, checks.mirror_lobe_db_vs_main);
    end
end
fprintf('G2/G3 %.1f s\n', toc(tStart));
end

function out = reorder(v, nIdx)
% values given in URA order -> sremu cell order
out = zeros(size(v));
out(nIdx + 1) = v;
end

function [x0, y0] = parabolic_peak(x, y, mask)
idx = find(mask);
[~, i] = max(y(idx));
i = idx(i);
a = y(i - 1); b = y(i); c = y(i + 1);
dlt = 0.5 * (a - c) / (a - 2 * b + c);
x0 = x(i) + dlt * (x(2) - x(1));
y0 = b - 0.25 * (a - c) * dlt;
end

function r = ifelse(c, a, b)
if c, r = a; else, r = b; end
end
