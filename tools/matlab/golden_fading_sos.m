function golden_fading_sos(outDir)
%GOLDEN_FADING_SOS  G6 golden: comm.RayleighChannel sum-of-sinusoids statistics (DESIGN.md
%   sections 5.11, 10.5 ST-02, 10.6). Flat Rayleigh, Jakes spectrum, f_D = 50 Hz,
%   f_s = 10 kHz, 48 sinusoids, 2e6 samples. Statistics of the RMS-normalized path gain:
%   complex autocorrelation for lags 0..100 ms, level-crossing rate and average fade
%   duration on a threshold grid (incl. -10 dB), envelope CDF on a dB grid.
%   Writes G6_fading_acf.csv, G6_fading_lcr.csv, G6_fading_cdf.csv and the meta file.
%   Accept (as ST-02): max |ACF - J0| <= 0.05 for lags <= 40 ms; LCR at -10 dB within 10 %
%   of 35.9 Hz; envelope KS < 0.01.
if nargin < 1, outDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'golden', 'matlab_R2026a'); end
tStart = tic;
fs = 10e3; fd = 50; nS = 2e6; nSin = 48;
ch = comm.RayleighChannel('SampleRate', fs, 'MaximumDopplerShift', fd, 'DopplerSpectrum', doppler('Jakes'), ...
    'PathGainsOutputPort', true, 'RandomStream', 'mt19937ar with seed', 'Seed', 12, ...
    'FadingTechnique', 'Sum of sinusoids', 'NumSinusoids', nSin);
[~, g] = ch(complex(ones(nS, 1)));
g = g(:);
p = mean(abs(g).^2);
gn = g / sqrt(p);
env = abs(gn);
maxLag = round(0.1 * fs);
gd = gn - mean(gn);
r = xcorr(gd, maxLag, 'unbiased'); r = r(maxLag + 1:end); r = r / r(1);
lags = (0:maxLag).' / fs;
j0 = besselj(0, 2 * pi * fd * lags);
thrDb = [(-30:2:6).'; -10];
thrDb = unique(thrDb);
lcr = zeros(size(thrDb)); afd = lcr;
for i = 1:numel(thrDb)
    rho = 10^(thrDb(i) / 20);
    below = env < rho;
    nDown = sum(~below(1:end-1) & below(2:end));
    lcr(i) = nDown / (nS / fs);
    afd(i) = mean(below) / max(lcr(i), eps);
end
levelDb = (-40:0.5:10).';
cdfv = arrayfun(@(x) mean(env <= 10^(x / 20)), levelDb);
o1 = golden_util_write_csv(fullfile(outDir, 'G6_fading_acf.csv'), {'lag_s', 'acf_re', 'acf_im', 'j0'}, ...
    {lags, real(r), imag(r), j0});
o2 = golden_util_write_csv(fullfile(outDir, 'G6_fading_lcr.csv'), {'threshold_db', 'lcr_hz', 'afd_s'}, {thrDb, lcr, afd});
o3 = golden_util_write_csv(fullfile(outDir, 'G6_fading_cdf.csv'), {'level_db', 'cdf'}, {levelDb, cdfv});
in40 = lags <= 0.040 + 1e-12;
rho10 = 10^(-10 / 20);
checks = struct('mean_power', p, 'max_abs_acf_minus_j0_40ms', max(abs(real(r(in40)) - j0(in40))), ...
    'lcr_m10db_hz', lcr(thrDb == -10), 'lcr_m10db_theory_hz', sqrt(2 * pi) * fd * rho10 * exp(-rho10^2), ...
    'ks_cdf_grid', max(abs(cdfv - (1 - exp(-(10.^(levelDb / 20)).^2)))));
params = struct('sample_rate_sps', fs, 'max_doppler_hz', fd, 'n_samples', nS, 'n_sinusoids', nSin, ...
    'seed', 12, 'random_stream', 'mt19937ar with seed', 'fading_technique', 'Sum of sinusoids', ...
    'normalization', 'path gain divided by sqrt(mean |g|^2)', 'acf', 'xcorr unbiased of the mean-removed gain, normalized to lag 0');
notes = {'comm.RayleighChannel SoS is the ACF/LCR reference; filtered Gaussian noise deviates from J0 by up to 0.127 (research/matlab_ris.md).'};
golden_util_write_meta(outDir, 'G6_fading_sos', mfilename, toc(tStart), params, checks, {o1, o2, o3}, notes);
fprintf('G6: max|ACF-J0| %.4f, LCR(-10 dB) %.2f Hz, KS %.4f, %.1f s\n', checks.max_abs_acf_minus_j0_40ms, ...
    checks.lcr_m10db_hz, checks.ks_cdf_grid, toc(tStart));
end
