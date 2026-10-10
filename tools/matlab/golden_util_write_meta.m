function golden_util_write_meta(outDir, goldenId, scriptName, runtime_s, params, checks, outputs, notes)
%GOLDEN_UTIL_WRITE_META  Write <goldenId>.meta.json next to the golden CSV files.
%   Records matlab_version, script and helper sha256 (LF-normalized), runtime_s, the
%   parameters, MATLAB-side self-checks and the sha256 of every output CSV
%   (DESIGN.md section 10.6).
here = fileparts(mfilename('fullpath'));
helpers = dir(fullfile(here, 'golden_util_*.m'));
hs = struct();
for i = 1:numel(helpers)
    key = strrep(helpers(i).name, '.', '_');
    hs.(key) = golden_util_sha256(fullfile(here, helpers(i).name));
end
meta = struct();
meta.golden_id = goldenId;
meta.data_class = 'MATLAB';
meta.matlab_version = version;
meta.matlab_release = version('-release');
meta.script = [scriptName '.m'];
meta.script_sha256 = golden_util_sha256(fullfile(here, [scriptName '.m']));
meta.helpers_sha256 = hs;
meta.runtime_s = runtime_s;
meta.created_utc = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));
meta.light_speed_mps = physconst('LightSpeed');
meta.parameters = params;
meta.checks = checks;
meta.outputs = outputs;
meta.notes = notes;
txt = jsonencode(meta, PrettyPrint=true);
fid = fopen(fullfile(outDir, [goldenId '.meta.json']), 'w');
fprintf(fid, '%s\n', txt);
fclose(fid);
end
