function info = golden_util_write_csv(path, names, cols)
%GOLDEN_UTIL_WRITE_CSV  Write a CSV with LF line endings and full double precision.
%   NAMES: 1xC cellstr of column names. COLS: 1xC cell; each entry is a numeric column
%   vector (written with %.17g, so doubles round-trip exactly) or a cellstr/string column.
%   Returns a struct with the file name, row count and sha256 (LF-normalized).
nc = numel(names);
assert(numel(cols) == nc, 'golden:csv', 'names/cols mismatch');
nr = numel(cols{1});
for c = 1:nc
    assert(numel(cols{c}) == nr, 'golden:csv', 'column %s has %d rows, expected %d', names{c}, numel(cols{c}), nr);
    if isstring(cols{c}), cols{c} = cellstr(cols{c}); end
end
fid = fopen(path, 'w');   % binary mode: '\n' stays LF on Windows
if fid < 0, error('golden:io', 'cannot open %s', path); end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', strjoin(names, ','));
for r = 1:nr
    parts = cell(1, nc);
    for c = 1:nc
        v = cols{c};
        if iscell(v)
            s = v{r};
            assert(~contains(s, ','), 'golden:csv', 'comma in string value');
            parts{c} = s;
        else
            parts{c} = sprintf('%.17g', v(r));
        end
    end
    fprintf(fid, '%s\n', strjoin(parts, ','));
end
clear cleanup
[~, f, e] = fileparts(path);
info = struct('file', [f e], 'rows', nr, 'sha256', golden_util_sha256(path));
end
