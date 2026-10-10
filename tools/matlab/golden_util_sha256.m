function h = golden_util_sha256(path)
%GOLDEN_UTIL_SHA256  SHA-256 (lower-case hex) of a text file with CRLF normalized to LF.
%   The Python side (tests/golden/harness.py) hashes the same way, so hashes do not
%   depend on how git checked the file out.
fid = fopen(path, 'r');
if fid < 0
    error('golden:io', 'cannot open %s', path);
end
b = fread(fid, inf, '*uint8');
fclose(fid);
b = b(:).';
if numel(b) > 1
    crlf = find(b(1:end-1) == 13 & b(2:end) == 10);
    b(crlf) = [];
end
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(typecast(b, 'int8'));
d = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(d, 2).', 1, []));
end
