function golden_thermal_noise(outDir)
%GOLDEN_THERMAL_NOISE  G5 golden: comm.ThermalNoise power (DESIGN.md sections 5.12, 10.6).
%   1e7 complex samples per configuration through comm.ThermalNoise with a zero input:
%     T290_1MSps   : 'Noise temperature' 290 K, 1 MS/s     (the PH-21 / G5 case)
%     NF7_1MSps    : 'Noise figure' 7 dB plus Add290KAntennaNoise, 1 MS/s (total k T0 F f_s)
%     T290_23p04MSps: 290 K at 23.04 MS/s
%   Records the measured mean power (W into the 1-ohm reference load, dBm) and the I/Q
%   variances. Accept: within +-0.01 dB of the emulator's k_B T0 F f_s noise (10^7 samples).
if nargin < 1, outDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'golden', 'matlab_R2026a'); end
tStart = tic;
rng(20261010, 'twister');
kB = physconst('Boltzmann');
nS = 1e7;
cfgs = { ...
    'T290_1MSps', 1e6, 0, {'NoiseMethod', 'Noise temperature', 'NoiseTemperature', 290}; ...
    'NF7_1MSps', 1e6, 7, {'NoiseMethod', 'Noise figure', 'NoiseFigure', 7, 'Add290KAntennaNoise', true}; ...
    'T290_23p04MSps', 23.04e6, 0, {'NoiseMethod', 'Noise temperature', 'NoiseTemperature', 290}};
name = cell(size(cfgs, 1), 1); fsV = zeros(size(name)); nfV = fsV; pW = fsV; pDbm = fsV;
vRe = fsV; vIm = fsV; mRe = fsV; mIm = fsV; aDbm = fsV;
for i = 1:size(cfgs, 1)
    [nm, fs, nf, args] = cfgs{i, :};
    tn = comm.ThermalNoise(args{:}, 'SampleRate', fs);
    y = tn(complex(zeros(nS, 1)));
    name{i} = nm; fsV(i) = fs; nfV(i) = nf;
    pW(i) = mean(abs(y).^2); pDbm(i) = 10 * log10(pW(i)) + 30;
    vRe(i) = var(real(y)); vIm(i) = var(imag(y)); mRe(i) = mean(real(y)); mIm(i) = mean(imag(y));
    aDbm(i) = 10 * log10(kB * 290 * 10^(nf / 10) * fs) + 30;
    fprintf('G5 %-15s %.4f dBm (analytic %.4f, diff %.4f dB)\n', nm, pDbm(i), aDbm(i), pDbm(i) - aDbm(i));
end
out = golden_util_write_csv(fullfile(outDir, 'G5_thermal_noise.csv'), ...
    {'config', 'sample_rate_sps', 'noise_figure_db', 'n_samples', 'mean_power_w', 'mean_power_dbm', ...
     'var_re_w', 'var_im_w', 'mean_re', 'mean_im', 'analytic_ktf_fs_dbm'}, ...
    {name, fsV, nfV, nS * ones(size(fsV)), pW, pDbm, vRe, vIm, mRe, mIm, aDbm});
params = struct('n_samples', nS, 'rng', 'rng(20261010, ''twister'')', 'boltzmann_j_per_k', kB, 'T0_k', 290, ...
    'input', 'complex zeros', 'reference_load_ohm', 1);
checks = struct('max_abs_measured_minus_analytic_db', max(abs(pDbm - aDbm)));
notes = {'Measured power is a 1e7-sample estimate (standard error about 0.0014 dB).'};
golden_util_write_meta(outDir, 'G5_thermal_noise', mfilename, toc(tStart), params, checks, {out}, notes);
fprintf('G5 %.1f s\n', toc(tStart));
end
