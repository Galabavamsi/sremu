function P = golden_util_panel(center, normal, uAxis, nU, nV, pitchU, pitchV, tilesU, tilesV, gapM)
%GOLDEN_UTIL_PANEL  Physical cells of a flat RIS panel (sremu scenario convention).
%   Cell order n = iv*nU + iu (0-based iu along uAxis, iv along v = normal x uAxis).
%   In-plane coordinates, centred on CENTER:
%     u = (iu - (nU-1)/2)*pitchU + (tu - (tilesU-1)/2)*gapM,  tu = floor(iu/(nU/tilesU))
%     v = (iv - (nV-1)/2)*pitchV + (tv - (tilesV-1)/2)*gapM,  tv = floor(iv/(nV/tilesV))
%   so the cell spacing across a tile boundary is pitch + gap (inert gap, DESIGN.md 5.3).
%   The quadrature area of every cell is its pitch area pitchU*pitchV.
n = normal(:) / norm(normal);
u = uAxis(:) - dot(uAxis(:), n) * n;
u = u / norm(u);
v = cross(n, u);
assert(mod(nU, tilesU) == 0 && mod(nV, tilesV) == 0, 'golden:panel', 'tiles must divide cells');
[iuG, ivG] = meshgrid(0:nU-1, 0:nV-1);       % rows iv, columns iu
iu = reshape(iuG.', 1, []);                    % n = iv*nU + iu  (iu fastest)
iv = reshape(ivG.', 1, []);
tu = floor(iu / (nU / tilesU));
tv = floor(iv / (nV / tilesV));
cu = (iu - (nU - 1) / 2) * pitchU + (tu - (tilesU - 1) / 2) * gapM;
cv = (iv - (nV - 1) / 2) * pitchV + (tv - (tilesV - 1) / 2) * gapM;
P.pos = center(:) + u * cu + v * cv;           % 3 x N
P.iu = iu; P.iv = iv; P.tu = tu; P.tv = tv;
P.area = pitchU * pitchV * ones(1, nU * nV);
P.axes = [n, u, v];                            % local x = normal (boresight), y = u, z = v
P.normal = n; P.u = u; P.v = v; P.center = center(:);
P.nU = nU; P.nV = nV; P.pitchU = pitchU; P.pitchV = pitchV;
P.tilesU = tilesU; P.tilesV = tilesV; P.gapM = gapM;
end
