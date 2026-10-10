function golden_cfo(outDir)
%GOLDEN_CFO  G4 golden: comm.PhaseFrequencyOffset on the reference TX IQ (DESIGN.md 10.6).
%   Input: G4_input_tx_iq.csv (2000 samples of RRC-filtered BPSK from
%   research/matlab/cfo_bpsk_crosscheck.m), rounded to single precision so that the
%   emulator's complex64 TX ring holds it exactly. Offsets -1400, -1375, +500, +750 and
%   +1400 Hz at f_s = 5880 S/s (the V-CFO rate): y[n] = x[n] exp(+j 2 pi df n / f_s).
%   Each offset is applied once over the whole vector and once in irregular chunks of one
%   System object (phase continuity across calls is recorded as a check).
%   Writes G4_cfo.csv and G4_cfo.meta.json. Accept: <= 1e-9 x RMS(y) for any chunking.
if nargin < 1, outDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'golden', 'matlab_R2026a'); end
tStart = tic;
inFile = fullfile(outDir, 'G4_input_tx_iq.csv');
T = readtable(inFile);
x = double(single(complex(T.tx_re, T.tx_im)));
fs = 5880;
offsets = [-1400 -1375 500 750 1400];
chunks = [1 7 128 333 1000];
names = {'n', 'x_re', 'x_im'};
cols = {T.n, real(x), imag(x)};
maxChunkDiff = 0;
for f0 = offsets
    pfo = comm.PhaseFrequencyOffset('FrequencyOffset', f0, 'SampleRate', fs, 'PhaseOffset', 0);
    y = pfo(x);
    pfo2 = comm.PhaseFrequencyOffset('FrequencyOffset', f0, 'SampleRate', fs, 'PhaseOffset', 0);
    yc = zeros(size(x)); i0 = 1; j = 0;
    while i0 <= numel(x)
        j = mod(j, numel(chunks)) + 1;
        idx = i0:min(i0 + chunks(j) - 1, numel(x));
        yc(idx) = pfo2(x(idx));
        i0 = idx(end) + 1;
    end
    maxChunkDiff = max(maxChunkDiff, max(abs(yc - y)) / sqrt(mean(abs(y).^2)));
    tag = strrep(strrep(sprintf('%+d', f0), '+', 'p'), '-', 'm');
    names = [names, {['y_' tag '_re'], ['y_' tag '_im']}]; %#ok<AGROW>
    cols = [cols, {real(y), imag(y)}]; %#ok<AGROW>
end
out = golden_util_write_csv(fullfile(outDir, 'G4_cfo.csv'), names, cols);
params = struct('f_s_sps', fs, 'offsets_hz', offsets, 'n_samples', numel(x), ...
    'input_file', 'G4_input_tx_iq.csv', 'input_sha256', golden_util_sha256(inFile), ...
    'input_rounding', 'double(single(x)) so that complex64 storage is exact', ...
    'convention', 'y[n] = x[n] exp(+j 2 pi df n / f_s), n from 0, PhaseOffset 0', 'chunk_sizes_cycled', chunks);
checks = struct('chunked_vs_whole_max_rel_rms', maxChunkDiff);
notes = {'The render contract applies lo_offset_hz = f_c,tx - f_c,rx as exp(+j 2 pi lo_offset t) at RX time t = n/f_s.'};
golden_util_write_meta(outDir, 'G4_cfo', mfilename, toc(tStart), params, checks, {out}, notes);
fprintf('G4: chunked vs whole %.2e (rel RMS), %.1f s\n', maxChunkDiff, toc(tStart));
end
