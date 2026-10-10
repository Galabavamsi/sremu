function golden_lab_polar(outDir)
%GOLDEN_LAB_POLAR  G1 golden: per-element near-field K1 sums (DESIGN.md section 10.6).
%   Lab-like 3.5 GHz geometries: layouts A_3x1, B_6x1 and C_3x2 of 5 x 8-cell tiles
%   {sigma1 sigma2 sigma1} (42.8 mm cells, 2 mm gaps: pitch 44.8 mm, extra 2 mm between
%   tiles), the 30-point polar grid of the ANTS Fig. 4(b) map (theta = 0,5,15,25,35,45 deg,
%   r = 0.6..1.8 m about the panel centre, theta from the panel normal), and three Tx
%   variants: T0_iso (3 m on the normal), T25_iso (3 m at -25 deg), T0_cos (T0 with a
%   CosPower(3) Tx and CosPower(1) Rx aimed at the panel centre).
%   Plus the Tang et al. 2022 Fig. 10 geometries: mmRIS1 (56 x 20 cells, 1.4 x 2.8 mm,
%   27 GHz, coding 0 = 0.9 at 165 deg), specular Tx/Rx at theta = 10/45 deg, d1 = 1/2 m,
%   d2 = 1..5 m, horns CosPower(53.8) aimed at the panel centre.
%   Writes G1_lab_polar.csv (one row per Tx/Rx pair), G1_cells.csv and G1_lab_polar.meta.json.
%   Accept (Python side): sremu vs golden max rel |dh| <= 1e-9.
if nargin < 1, outDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'golden', 'matlab_R2026a'); end
if ~exist(outDir, 'dir'), mkdir(outDir); end
tStart = tic;

iso = struct('alpha', NaN, 'boresight', [1; 0; 0]);
rows = struct('case_id', {}, 'geometry', {}, 'variant', {}, 'fc', {}, 'tx', {}, 'rx', {}, ...
    'tx_ant', {}, 'rx_ant', {}, 'h_los', {}, 'h_ris', {}, 'h_hand', {});
cellRows = struct('geometry', {}, 'n', {}, 'iu', {}, 'iv', {}, 'pos', {}, 'state', {});
chkCos = 0; chkRange = 0; maxToolboxVsHand = 0;
geomMeta = struct();

% ------------------------------------------------------------------ lab layouts (3.5 GHz)
fcLab = 3.5e9;
pitch = 0.0448; gap = 0.002;
alphabetLab = [exp(1j * deg2rad(49.64)), exp(1j * deg2rad(-153.77))];   % assume_mag 1.0
layouts = { ...
    'lab_A_3x1', 24, 5, 3, 1, [0 1 0]; ...
    'lab_B_6x1', 48, 5, 6, 1, [0 0 1 1 0 0]; ...
    'lab_C_3x2', 24, 10, 3, 2, [0 1 0; 0 1 0]};
thetas = [0 5 15 25 35 45];
radii = [0.6 0.9 1.2 1.5 1.8];
caseId = 0;
for L = 1:size(layouts, 1)
    [name, nU, nV, tU, tV, tileState] = layouts{L, :};
    P = golden_util_panel([0 0 0], [1 0 0], [0 1 0], nU, nV, pitch, pitch, tU, tV, gap);
    state = tileState(sub2ind(size(tileState), P.tv + 1, P.tu + 1));
    gam = alphabetLab(state + 1);
    for n = 1:numel(state)
        cellRows(end + 1) = struct('geometry', name, 'n', n - 1, 'iu', P.iu(n), 'iv', P.iv(n), ...
            'pos', P.pos(:, n), 'state', state(n)); %#ok<AGROW>
    end
    geomMeta.(name) = struct('fc_hz', fcLab, 'center_m', [0 0 0], 'normal', [1 0 0], 'u_axis', [0 1 0], ...
        'n_u', nU, 'n_v', nV, 'pitch_u_m', pitch, 'pitch_v_m', pitch, 'n_tiles_u', tU, 'n_tiles_v', tV, ...
        'gap_m', gap, 'tile_states', tileState, 'alphabet_mag', abs(alphabetLab), ...
        'alphabet_phase_deg', [49.64 -153.77], 'cell_model', 'point');
    variants = { ...
        'T0_iso', [3; 0; 0], iso, 'iso', NaN; ...
        'T25_iso', 3 * [cosd(-25); sind(-25); 0], iso, 'iso', NaN; ...
        'T0_cos', [3; 0; 0], struct('alpha', 3, 'boresight', [-1; 0; 0]), 'cos3', 1};
    for V = 1:size(variants, 1)
        [vname, tx, txAnt, txLabel, rxAlpha] = variants{V, :};
        for th = thetas
            for r = radii
                rx = r * [cosd(th); sind(th); 0];
                if isnan(rxAlpha)
                    rxAnt = iso; rxLabel = 'iso';
                else
                    rxAnt = struct('alpha', rxAlpha, 'boresight', -rx / norm(rx)); rxLabel = 'cos1';
                end
                [h, hh, hl, chk] = golden_util_k1(tx, rx, P, fcLab, gam, txAnt, rxAnt);
                caseId = caseId + 1;
                rows(end + 1) = struct('case_id', caseId, 'geometry', name, 'variant', vname, 'fc', fcLab, ...
                    'tx', tx, 'rx', rx, 'tx_ant', txLabel, 'rx_ant', rxLabel, 'h_los', hl, 'h_ris', h, 'h_hand', hh); %#ok<AGROW>
                chkCos = max(chkCos, chk.cosine_element_max_dev);
                chkRange = max(chkRange, chk.range_max_dev_m);
                maxToolboxVsHand = max(maxToolboxVsHand, abs(h - hh) / abs(hh));
            end
        end
    end
end

% ------------------------------------------------------------------ Tang 2022 Fig. 10 (27 GHz)
fcT = 27e9;
alphaHorn = 53.8;
gamTang = 0.9 * exp(1j * deg2rad(165));
P = golden_util_panel([0 0 0], [0 0 1], [1 0 0], 56, 20, 1.4e-3, 2.8e-3, 1, 1, 0);
for n = 1:size(P.pos, 2)
    cellRows(end + 1) = struct('geometry', 'tang_mmris1', 'n', n - 1, 'iu', P.iu(n), 'iv', P.iv(n), ...
        'pos', P.pos(:, n), 'state', 0); %#ok<AGROW>
end
geomMeta.tang_mmris1 = struct('fc_hz', fcT, 'center_m', [0 0 0], 'normal', [0 0 1], 'u_axis', [1 0 0], ...
    'n_u', 56, 'n_v', 20, 'pitch_u_m', 1.4e-3, 'pitch_v_m', 2.8e-3, 'n_tiles_u', 1, 'n_tiles_v', 1, ...
    'gap_m', 0, 'alphabet_mag', [0.9 0.7], 'alphabet_phase_deg', [165 0], 'state_all_cells', 0, ...
    'horn_cos_power_alpha', alphaHorn, 'cell_model', 'point');
panels = {'a', 1.0, 10; 'b', 2.0, 10; 'c', 1.0, 45; 'd', 2.0, 45};
for Q = 1:size(panels, 1)
    [pname, d1, th] = panels{Q, :};
    tx = d1 * [-sind(th); 0; cosd(th)];
    txAnt = struct('alpha', alphaHorn, 'boresight', -tx / norm(tx));
    for d2 = 1.0:0.5:5.0
        rx = d2 * [sind(th); 0; cosd(th)];
        rxAnt = struct('alpha', alphaHorn, 'boresight', -rx / norm(rx));
        [h, hh, hl, chk] = golden_util_k1(tx, rx, P, fcT, gamTang * ones(1, size(P.pos, 2)), txAnt, rxAnt);
        caseId = caseId + 1;
        rows(end + 1) = struct('case_id', caseId, 'geometry', 'tang_mmris1', 'variant', ['fig10' pname], ...
            'fc', fcT, 'tx', tx, 'rx', rx, 'tx_ant', 'cos53.8', 'rx_ant', 'cos53.8', ...
            'h_los', hl, 'h_ris', h, 'h_hand', hh); %#ok<AGROW>
        chkCos = max(chkCos, chk.cosine_element_max_dev);
        chkRange = max(chkRange, chk.range_max_dev_m);
        maxToolboxVsHand = max(maxToolboxVsHand, abs(h - hh) / abs(hh));
    end
end

% ------------------------------------------------------------------ write
txM = [rows.tx]; rxM = [rows.rx];
hl = [rows.h_los]; hr = [rows.h_ris]; hh = [rows.h_hand];
out1 = golden_util_write_csv(fullfile(outDir, 'G1_lab_polar.csv'), ...
    {'case_id', 'geometry', 'variant', 'fc_hz', 'tx_x', 'tx_y', 'tx_z', 'rx_x', 'rx_y', 'rx_z', ...
     'tx_antenna', 'rx_antenna', 'h_los_re', 'h_los_im', 'h_ris_re', 'h_ris_im', 'h_ris_hand_re', 'h_ris_hand_im'}, ...
    {[rows.case_id].', {rows.geometry}.', {rows.variant}.', [rows.fc].', txM(1, :).', txM(2, :).', txM(3, :).', ...
     rxM(1, :).', rxM(2, :).', rxM(3, :).', {rows.tx_ant}.', {rows.rx_ant}.', real(hl).', imag(hl).', ...
     real(hr).', imag(hr).', real(hh).', imag(hh).'});
posM = [cellRows.pos];
out2 = golden_util_write_csv(fullfile(outDir, 'G1_cells.csv'), ...
    {'geometry', 'n', 'iu', 'iv', 'x', 'y', 'z', 'state'}, ...
    {{cellRows.geometry}.', [cellRows.n].', [cellRows.iu].', [cellRows.iv].', posM(1, :).', posM(2, :).', ...
     posM(3, :).', [cellRows.state].'});

params = struct('geometries', geomMeta, 'polar_theta_deg', thetas, 'polar_r_m', radii, ...
    'tx_variants', struct('T0_iso', '[3,0,0] isotropic Tx and Rx', 'T25_iso', '3*[cos(-25 deg), sin(-25 deg), 0] isotropic', ...
        'T0_cos', '[3,0,0]; Tx CosPower(3) and Rx CosPower(1), boresights toward the panel centre'), ...
    'tang_panels', struct('a', [1 10], 'b', [2 10], 'c', [1 45], 'd', [2 45]), 'tang_d2_m', 1.0:0.5:5.0, ...
    'kernel', 'K1 point: sqrt(GT GR) Gamma A sqrt(cos_i cos_s)/(4 pi r_i r_s) j exp(-jk(r_i+r_s)), visibility mask', ...
    'antenna_cos_power', 'G = 2(alpha+1) cos^alpha(psi): phased.CosineAntennaElement CosinePower [alpha/2 alpha/2] (voltage pattern) times sqrt(G0)');
checks = struct('cosine_element_field_pattern_max_dev', chkCos, 'rangeangle_vs_vecnorm_max_dev_m', chkRange, ...
    'toolbox_vs_hand_max_rel', maxToolboxVsHand, 'n_rows', numel(rows));
notes = {'h_* columns are full complex-baseband coefficients at the carrier (propagation phase included).', ...
    'Cell order n = iv*n_u + iu; G1_cells.csv lists every cell position and state.', ...
    'MATLAB agreement is code-to-code only (DESIGN.md section 1.6).'};
golden_util_write_meta(outDir, 'G1_lab_polar', mfilename, toc(tStart), params, checks, {out1, out2}, notes);
fprintf('G1: %d rows, toolbox vs hand max rel %.3e, cosine element dev %.3e, %.1f s\n', ...
    numel(rows), maxToolboxVsHand, chkCos, toc(tStart));
end
