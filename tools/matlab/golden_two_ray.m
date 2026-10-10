function golden_two_ray(outDir)
%GOLDEN_TWO_RAY  G8 golden: flat-earth two-ray channel at 100 MHz (DESIGN.md 5.7, 10.6).
%   Uses twoRayChannel (R2026a name of phased.TwoRayChannel, Radar Toolbox; properties checked
%   with 'help': OperatingFrequency, SampleRate, GroundReflectionCoefficient,
%   CombinedRaysOutput). Tx at height 200 m, Rx at 1.5 m and 10 m, ground range 48, 52 and
%   56 km, Gamma = -1, isotropic antennas. A constant (DC) input is propagated; once both
%   rays have arrived the output equals the complex two-ray gain. Path loss = -20 log10|h|.
%   Writes G8_two_ray.csv and G8_two_ray.meta.json. Accept: +-0.05 dB.
if nargin < 1, outDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'golden', 'matlab_R2026a'); end
tStart = tic;
c0 = physconst('LightSpeed');
fc = 100e6; fs = 1e6; lam = c0 / fc; k = 2 * pi / lam;
ht = 200;
dKm = [48 52 56]; hrs = [1.5 10];
rows = zeros(numel(dKm) * numel(hrs), 7); i = 0;
for d = dKm * 1e3
    for hr = hrs
        ch = twoRayChannel('OperatingFrequency', fc, 'SampleRate', fs, 'GroundReflectionCoefficient', -1, ...
            'CombinedRaysOutput', true, 'MaximumDistanceSource', 'Property', 'MaximumDistance', 1.2 * d);
        nIn = ceil(2 * d / c0 * fs) + 200;
        y = ch(ones(nIn, 1), [0; 0; ht], [d; 0; hr], [0; 0; 0], [0; 0; 0]);
        h = y(end);
        d0 = hypot(d, ht - hr); d1 = hypot(d, ht + hr);
        hHand = lam / (4 * pi) * (exp(-1j * k * d0) / d0 - exp(-1j * k * d1) / d1);
        i = i + 1;
        rows(i, :) = [d, ht, hr, real(h), imag(h), -20 * log10(abs(h)), -20 * log10(abs(hHand))];
        fprintf('G8 d = %5.1f km hr = %4.1f m: PL %.4f dB (hand %.4f dB)\n', d / 1e3, hr, rows(i, 6), rows(i, 7));
    end
end
out = golden_util_write_csv(fullfile(outDir, 'G8_two_ray.csv'), ...
    {'d_m', 'ht_m', 'hr_m', 'h_re', 'h_im', 'path_loss_db', 'hand_two_ray_path_loss_db'}, ...
    {rows(:, 1), rows(:, 2), rows(:, 3), rows(:, 4), rows(:, 5), rows(:, 6), rows(:, 7)});
params = struct('fc_hz', fc, 'sample_rate_sps', fs, 'ground_reflection_coefficient', -1, 'ht_m', ht, ...
    'ground_range_m', dKm * 1e3, 'hr_m', hrs, 'object', 'twoRayChannel (phased.TwoRayChannel renamed in R2026a)', ...
    'input', 'ones; output sample taken after both rays arrived');
checks = struct('max_abs_toolbox_minus_hand_db', max(abs(rows(:, 6) - rows(:, 7))));
notes = {'Phase of h follows the toolbox convention; only |h| (path loss) is compared (+-0.05 dB).'};
golden_util_write_meta(outDir, 'G8_two_ray', mfilename, toc(tStart), params, checks, {out}, notes);
fprintf('G8 %.1f s\n', toc(tStart));
end
