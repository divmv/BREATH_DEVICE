%% Diagnose_XYP_Artifact.m
% End-to-end software diagnostics for X-Y-Power scans (laser + photodiode array).
% Assumes CSV columns: [Samples, Power, X, Y]  (1=Samples, 2=Power, 3=X, 4=Y)

clc; clear; close all;

%% =================== CONFIG ===================
FILE = '';  % e.g., 'C:\Users\Alex\Dice_lab\captures\analog_20251014_125119.csv'
Fs   = 1000;                % sampling rate [Hz] (for labels only)
SMOOTH_WINS = [0 50 100 300 600 1200];  % 0 = raw; others are movmean windows (samples)
LAG_RANGE   = -60:60;       % sample shifts to test for X/Y vs Power misalignment
GRID_BINS   = 100;          % heatmap resolution (bins per axis)
POINT_SIZE  = 4;            % scatter marker size
USE_FLAT    = false;        % set true if you have a flat-field CSV
FLAT_FILE   = '';           % path to flat CSV (same geometry), if USE_FLAT = true
%% ==============================================

% -------- File selection if not specified --------
if isempty(FILE)
    [f, p] = uigetfile({'*.csv'}, 'Select data CSV'); if isequal(f,0), return; end
    FILE = fullfile(p,f);
end
fprintf('Loading: %s\n', FILE);

% -------- Load & sanity checks --------
md = readmatrix(FILE);
assert(size(md,2) >= 4, 'Expected at least 4 columns: [Samples, Power, X, Y].');
S  = md(:,1);
P0 = md(:,2);     % RAW Power
X  = md(:,3);
Y  = md(:,4);

if ~(issorted(S) && all(diff(S) >= 0))
    warning('Column 1 (Samples) is not strictly monotonic. Proceeding anyway.');
end

%% ---------- RAW 3D scatter ----------
fig1 = figure('Name','RAW 3D X-Y-Power','Color','w');
scatter3(X, Y, P0, POINT_SIZE, P0, '.'); grid on; box on
title(sprintf('3D - XYP (RAW Power)  |  Fs=%g Hz', Fs));
xlabel('X'); ylabel('Y'); zlabel('Power'); view(135,20)

%% ---------- Smoothing sweep ----------
fig2 = figure('Name','Smoothing Sweep','Color','w');
tiledlayout('flow','Padding','compact','TileSpacing','compact');
for w = SMOOTH_WINS
    nexttile;
    if w > 0
        P = movmean(P0, w, 'Endpoints','shrink');
        ttl = sprintf('movmean window = %d', w);
    else
        P = P0;
        ttl = 'RAW (no smoothing)';
    end
    scatter3(X, Y, P, POINT_SIZE, P, '.'); grid on; box on
    title(ttl); xlabel('X'); ylabel('Y'); zlabel('Power'); view(135,20)
end

%% ---------- Lag search (Power vs X/Y) ----------
bestLag = 0; bestScore = inf; scores = nan(size(LAG_RANGE));
for ii = 1:numel(LAG_RANGE)
    k  = LAG_RANGE(ii);
    Pk = circshift(P0, k);
    [score, ~, xb, yb] = medianGridScore(X, Y, Pk, GRID_BINS);
    scores(ii) = score;
    if score < bestScore
        bestScore = score;
        bestLag   = k;
    end
end
fprintf('Best lag (Power shift) = %d samples (lower VAR score is better)\n', bestLag);

fig3 = figure('Name','Lag Search','Color','w');
subplot(2,1,1);
plot(LAG_RANGE, scores, '-o','LineWidth',1.2); grid on; box on
xlabel('Power shift (samples)'); ylabel('Score = var(median grid)');
title(sprintf('Best shift = %d   |   min score = %.4g', bestLag, min(scores)));

subplot(2,1,2);
Pk = circshift(P0, bestLag);
scatter3(X, Y, Pk, POINT_SIZE, Pk, '.'); grid on; box on
title(sprintf('3D - XYP with Power shifted by %d samples', bestLag));
xlabel('X'); ylabel('Y'); zlabel('Power'); view(135,20)

%% ---------- Median heatmaps (order-agnostic) ----------
[~, Graw, xb1, yb1] = medianGridScore(X, Y, P0, GRID_BINS);
[~, Gshf, xb2, yb2] = medianGridScore(X, Y, Pk, GRID_BINS);

fig4 = figure('Name','Median Power Heatmaps','Color','w');
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile; imagesc(xb1(1:end-1), yb1(1:end-1), Graw'); axis xy; colorbar
title('Median Power Heatmap (RAW)'); xlabel('X'); ylabel('Y');
nexttile; imagesc(xb2(1:end-1), yb2(1:end-1), Gshf'); axis xy; colorbar
title(sprintf('Median Power Heatmap (shift = %d)', bestLag)); xlabel('X'); ylabel('Y');

%% ---------- Histogram (clipping/compression check) ----------
fig5 = figure('Name','Power Histogram','Color','w');
edges = linspace(min(P0), max(P0), 100);
counts = histcounts(P0, edges);
stairs(edges(1:end-1), counts, 'LineWidth',1.5); grid on; box on
title('Power histogram (RAW)'); xlabel('Power'); ylabel('Count');

if any(P0 == max(P0)) || any(P0 == min(P0))
    warning('Power hits min or max value (possible clipping/compression).');
end

%% ---------- Optional: Flat-field correction ----------
if USE_FLAT
    if isempty(FLAT_FILE)
        [ff, pp] = uigetfile({'*.csv'}, 'Select flat-field CSV'); 
        if isequal(ff,0), USE_FLAT = false; 
        else, FLAT_FILE = fullfile(pp,ff);
        end
    end
end

if USE_FLAT
    fprintf('Loading flat-field: %s\n', FLAT_FILE);
    md_flat = readmatrix(FLAT_FILE);
    assert(size(md_flat,2) >= 4, 'Flat CSV must match format [Samples, Power, X, Y].');
    Pf = md_flat(:,2);
    Xf = md_flat(:,3);
    Yf = md_flat(:,4);

    % Build flat gain map
    [~, Gflat, xbf, ybf] = medianGridScore(Xf, Yf, Pf, GRID_BINS);
    % Interpolate flat to sample points
    Gi = interp2(xbf(1:end-1), ybf(1:end-1), Gflat', X, Y, 'linear', NaN);
    Pcorr = P0 ./ Gi;

    fig6 = figure('Name','Flat-field correction','Color','w');
    tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
    nexttile; imagesc(xbf(1:end-1), ybf(1:end-1), Gflat'); axis xy; colorbar
    title('Flat gain map'); xlabel('X'); ylabel('Y');
    nexttile; scatter3(X,Y,P0,POINT_SIZE,P0,'.'); grid on; box on
    title('Before correction'); xlabel('X'); ylabel('Y'); zlabel('Power'); view(135,20)
    nexttile; scatter3(X,Y,Pcorr,POINT_SIZE,Pcorr,'.'); grid on; box on
    title('After flat-field correction'); xlabel('X'); ylabel('Y'); zlabel('Power'); view(135,20)
end

%% ---------- Console summary ----------
fprintf('\n==== Summary ====\n');
fprintf('File: %s\n', FILE);
fprintf('Best lag (Power shift): %d samples\n', bestLag);
fprintf('VAR score RAW heatmap : %.4g\n', nanvar(Graw(:)));
fprintf('VAR score SHIFT heatmap: %.4g\n', nanvar(Gshf(:)));
if USE_FLAT
    fprintf('Flat-field applied: %s\n', FLAT_FILE);
end
fprintf(['Interpretation:\n' ...
         ' • If best lag ≠ 0 and visuals improve → timing/misalignment in software pipeline.\n' ...
         ' • If smaller smoothing windows change little → not a movmean artifact.\n' ...
         ' • If heatmap hole persists after best lag → artifact is in the data (optics/sensor/AFE).\n' ...
         ' • Histogram pile-up at extremes → clipping/compression (reduce power or gain).\n']);
fprintf('=================\n');

%% ======== Local helper (no external deps) ========
function [score, G, xb, yb] = medianGridScore(X, Y, P, nbins)
    % Bin (X,Y) to a regular grid and take median Power per bin.
    % Returns score = variance of the median grid (ignoring NaNs).
    [~, xb, yb, binX, binY] = histcounts2(X, Y, nbins, nbins);
    valid = ~isnan(binX) & ~isnan(binY);
    G = accumarray([binX(valid) binY(valid)], P(valid), [numel(xb)-1 numel(yb)-1], @median, NaN);
    score = nanvar(G(:));
end
