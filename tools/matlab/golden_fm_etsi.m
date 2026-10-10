function golden_fm_etsi(outDir)
%GOLDEN_FM_ETSI  G7 golden: focused RIS at 100 MHz vs the ETSI far-field closed form
%   (DESIGN.md section 10.6). Geometry of the v2 FM figure: Tx at the origin; 80 x 80 cells
%   of 0.5 m (0.167 lambda) centred at (0, 0, 80 km) facing -z (u_axis = +x, v = n x u = -y);
%   cells focused on (0, 0, 78 km) with K1 phases exp(+jk(r_i + r_s)) * (-j), quantized to an
%   ideal 4-bit alphabet (16 states); Rx on the z axis at 1..79 km (1 km steps).
%   Per Rx: LOS and K1 coefficients (toolbox path), the PO bound A/(4 pi d_t d_r) of a perfectly
%   phased plate of the same area, and the ETSI GR RIS 003 far-field value at the focus
%   N dx dy/(4 pi d_t d_r). Accept: K1 at the focus within 1 dB of ETSI and never above the
%   PO bound (0.01 dB).
if nargin < 1, outDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'golden', 'matlab_R2026a'); end
tStart = tic;
c0 = physconst('LightSpeed');
fc = 100e6; lam = c0 / fc; k = 2 * pi / lam;
nU = 80; nV = 80; pitch = 0.5; nBits = 4;
center = [0 0 80e3];
P = golden_util_panel(center, [0 0 -1], [1 0 0], nU, nV, pitch, pitch, 1, 1, 0);
tx = [0; 0; 0]; focus = [0; 0; 78e3];
ri = vecnorm(tx - P.pos); rs = vecnorm(focus - P.pos);
wIdeal = exp(1j * k * (ri + rs)) * (-1j);
s = mod(round(angle(wIdeal) / (2 * pi / 2^nBits)), 2^nBits);
gam = exp(1j * 2 * pi * s / 2^nBits);
iso = struct('alpha', NaN, 'boresight', [1; 0; 0]);
zs = (1e3:1e3:79e3);
hl = zeros(size(zs)); hr = hl; hh = hl; bound = hl; dT = norm(center(:) - tx);
for i = 1:numel(zs)
    rx = [0; 0; zs(i)];
    [hr(i), hh(i), hl(i)] = golden_util_k1(tx, rx, P, fc, gam, iso, iso);
    bound(i) = nU * nV * pitch^2 / (4 * pi * dT * norm(rx - center(:)));
end
iF = find(zs == 78e3);
etsi = nU * nV * pitch * pitch / (4 * pi * dT * norm(focus - center(:)));
o1 = golden_util_write_csv(fullfile(outDir, 'G7_fm_etsi.csv'), ...
    {'rx_z_m', 'h_los_re', 'h_los_im', 'h_ris_re', 'h_ris_im', 'h_ris_hand_re', 'h_ris_hand_im', 'po_bound_abs'}, ...
    {zs.', real(hl).', imag(hl).', real(hr).', imag(hr).', real(hh).', imag(hh).', bound.'});
o2 = golden_util_write_csv(fullfile(outDir, 'G7_cells.csv'), {'n', 'iu', 'iv', 'x', 'y', 'z', 'state'}, ...
    {(0:nU*nV-1).', P.iu.', P.iv.', P.pos(1, :).', P.pos(2, :).', P.pos(3, :).', s.'});
checks = struct('k1_focus_db', 20 * log10(abs(hr(iF))), 'etsi_focus_db', 20 * log10(etsi), ...
    'k1_minus_etsi_db_at_focus', 20 * log10(abs(hr(iF)) / etsi), ...
    'max_k1_minus_po_bound_db', max(20 * log10(abs(hr) ./ bound)), ...
    'toolbox_vs_hand_max_rel', max(abs(hr - hh) ./ abs(hh)));
params = struct('fc_hz', fc, 'panel', struct('center_m', center, 'normal', [0 0 -1], 'u_axis', [1 0 0], ...
    'n_u', nU, 'n_v', nV, 'pitch_u_m', pitch, 'pitch_v_m', pitch, 'cell_model', 'point'), ...
    'tx_m', tx.', 'focus_m', focus.', 'rx_z_m', [zs(1) zs(end) 1e3], 'n_bits', nBits, ...
    'alphabet', sprintf('ideal %d-bit exp(j 2 pi s/%d)', nBits, 2^nBits), ...
    'etsi_focus_abs', etsi, 'focus_row_index_0based', iF - 1, 'd_t_m', dT);
notes = {'Rx lies between the Tx and the panel on the panel axis; h_* include the propagation phase.', ...
    'ETSI closed form with A = G_t = G_r = 1 and a unit cell pattern; MATLAB agreement is code-to-code only.'};
golden_util_write_meta(outDir, 'G7_fm_etsi', mfilename, toc(tStart), params, checks, {o1, o2}, notes);
fprintf('G7: K1 - ETSI at focus %.3f dB; max K1 - PO bound %.3f dB; %.1f s\n', checks.k1_minus_etsi_db_at_focus, ...
    checks.max_k1_minus_po_bound_db, toc(tStart));
end
