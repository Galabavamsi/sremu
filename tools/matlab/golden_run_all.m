function golden_run_all(outDir, which)
%GOLDEN_RUN_ALL  Regenerate the MATLAB goldens G1-G8 (DESIGN.md section 10.6).
%   golden_run_all(outDir) runs every golden script; golden_run_all(outDir, {'G4','G8'})
%   runs a subset. Called by run_goldens.ps1 through MATLAB R2026a -batch. Exits with an
%   error (non-zero status under -batch) if any script fails or the release is not 2026a.
if nargin < 1 || isempty(outDir)
    outDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'tests', 'golden', 'matlab_R2026a');
end
if nargin < 2, which = {}; end
if ~strcmp(version('-release'), '2026a')
    error('golden:release', 'goldens must be generated with MATLAB R2026a, this is %s', version('-release'));
end
if ~exist(outDir, 'dir'), mkdir(outDir); end
scripts = { ...
    'G1', @golden_lab_polar; ...
    'G2', @golden_beam_pattern; ...   % also writes G3
    'G4', @golden_cfo; ...
    'G5', @golden_thermal_noise; ...
    'G6', @golden_fading_sos; ...
    'G7', @golden_fm_etsi; ...
    'G8', @golden_two_ray};
failed = {};
tAll = tic;
for i = 1:size(scripts, 1)
    id = scripts{i, 1};
    if ~isempty(which) && ~any(strcmp(which, id)) && ~(strcmp(id, 'G2') && any(strcmp(which, 'G3')))
        continue
    end
    fprintf('==== %s (%s)\n', id, func2str(scripts{i, 2}));
    try
        scripts{i, 2}(outDir);
    catch ME
        fprintf(2, '%s FAILED: %s\n', id, getReport(ME, 'extended', 'hyperlinks', 'off'));
        failed{end + 1} = id; %#ok<AGROW>
    end
end
fprintf('goldens done in %.1f s\n', toc(tAll));
if ~isempty(failed)
    error('golden:failed', 'failed goldens: %s', strjoin(failed, ', '));
end
end
