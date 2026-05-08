% Run unified Bayesian SMM tests for smoothing, prediction, and control.
%
% The task cfg values below follow the standalone smooth/predict/control
% numerical settings used in the paper examples, then call the unified MM
% comparison script once per task and summarize estimation errors.
%
% Optional quick run:
%   testOptions = struct('Ne', 1, 'mmMaxIter', 1, 'sqpMaxIter', 1, ...
%       'optimMaxIterations', 50, 'optimMaxFunctionEvaluations', 2000, ...
%       'saveSummary', false, 'saveFigure', false);
%   run('bayesSMMTest.m')

clc; close all;

if ~exist('testOptions', 'var') || isempty(testOptions)
    testOptions = default_bayes_smm_test_options();
else
    testOptions = merge_test_options(default_bayes_smm_test_options(), testOptions);
end

if (testOptions.saveSummary || testOptions.saveFigure || testOptions.saveUnifiedResults) ...
        && ~exist(testOptions.outputDir, 'dir')
    mkdir(testOptions.outputDir);
end
if exist(fullfile(pwd, 'boxplotGroup'), 'dir')
    addpath(fullfile(pwd, 'boxplotGroup'));
end

taskSpecs = make_bayes_smm_task_specs(testOptions);
numTasks = numel(taskSpecs);
taskResults = cell(1, numTasks);

for tt = 1:numTasks
    fprintf('\n=== Running %s task (%d/%d) ===\n', ...
        taskSpecs(tt).label, tt, numTasks);
    taskResults{tt} = run_unified_bayes_smm_task(taskSpecs(tt).cfg);
end

[errnorm, methodLabels, taskLabels] = collect_task_errors(taskResults, taskSpecs);
fig = plot_task_error_comparison(errnorm, methodLabels, taskLabels);

if testOptions.saveSummary
    summaryFile = fullfile(testOptions.outputDir, 'bayesSMMTest_unified_results.mat');
    save(summaryFile, 'taskResults', 'errnorm', 'methodLabels', ...
        'taskLabels', 'testOptions');
    fprintf('\nSaved summary: %s\n', summaryFile);
end

if testOptions.saveFigure
    figFile = fullfile(testOptions.outputDir, 'bayesSMMTest_estimation_error_boxplot.fig');
    pngFile = fullfile(testOptions.outputDir, 'bayesSMMTest_estimation_error_boxplot.png');
    savefig(fig, figFile);
    try
        exportgraphics(fig, pngFile, 'Resolution', 300);
    catch
        saveas(fig, pngFile);
    end
    fprintf('Saved figure: %s\n', pngFile);
end

%% Local helpers
function opts = default_bayes_smm_test_options()
    opts = struct();
    opts.Ne = 100;
    opts.seedSmooth = 1;
    opts.seedPredict = 1;
    opts.seedControl = 1;
    opts.mmMaxIter = 100;
    opts.mmTol = 1e-3;
    opts.mmEta = 1e-4;
    opts.sqpMaxIter = 100;
    opts.sqpTol = 1e-6;
    opts.optimMaxIterations = 1e4;
    opts.optimMaxFunctionEvaluations = 1e5;
    opts.saveUnifiedResults = false;
    opts.saveSummary = true;
    opts.saveFigure = true;
    opts.outputDir = fullfile(pwd, 'data');
end

function out = merge_test_options(defaults, overrides)
    out = defaults;
    names = fieldnames(overrides);
    for ii = 1:numel(names)
        out.(names{ii}) = overrides.(names{ii});
    end
end

function taskSpecs = make_bayes_smm_task_specs(opts)
    taskSpecs = repmat(struct('label', '', 'cfg', struct()), 1, 3);

    cfg = base_unified_cfg(opts);
    cfg.task = 'smooth';
    cfg.experiment.seed = opts.seedSmooth;
    cfg.system.model = 'random';
    cfg.noise.distribution = 'studentT';
    cfg.noise.inputNoise = false;
    cfg.noise.dof = 10;
    cfg.noise.correlation = 0;
    cfg.noise.u_data_var = 0;
    cfg.noise.u_var = 0;
    cfg.noise.y_data_var = 1e-4;
    cfg.noise.y_var = 1e-2;
    taskSpecs(1).label = 'Smoothing';
    taskSpecs(1).cfg = cfg;

    cfg = base_unified_cfg(opts);
    cfg.task = 'predict';
    cfg.experiment.seed = opts.seedPredict;
    cfg.system.model = 'random';
    cfg.noise.distribution = 'gaussian';
    cfg.noise.inputNoise = true;
    cfg.noise.correlation = 0;
    cfg.noise.u_data_var = 1e-4;
    cfg.noise.y_data_var = 1e-4;
    cfg.noise.u_var = 1e-2;
    cfg.noise.y_var = 1e-2;
    taskSpecs(2).label = 'Prediction';
    taskSpecs(2).cfg = cfg;

    cfg = base_unified_cfg(opts);
    cfg.task = 'control';
    cfg.experiment.seed = opts.seedControl;
    cfg.system.model = 'controlToeplitz';
    cfg.noise.distribution = 'gaussian';
    cfg.noise.inputNoise = false;
    cfg.noise.correlation = 0.95;
    cfg.noise.u_data_var = 0;
    cfg.noise.u_var = 0;
    cfg.noise.y_data_var = 1e-2;
    cfg.noise.y_var = 1e-2;
    cfg.control.q = 5;
    cfg.control.r = 0.5;
    cfg.control.referencePattern = [1; -1; 1];
    taskSpecs(3).label = 'Control';
    taskSpecs(3).cfg = cfg;
end

function cfg = base_unified_cfg(opts)
    cfg = struct();
    cfg.experiment.Ne = opts.Ne;
    cfg.data.N = 100;
    cfg.data.L = 40;
    cfg.system.nx = 10;
    cfg.system.nu = 1;
    cfg.system.ny = 1;
    cfg.mm.maxIter = opts.mmMaxIter;
    cfg.mm.tol = opts.mmTol;
    cfg.mm.eta = opts.mmEta;
    cfg.sqp.maxIter = opts.sqpMaxIter;
    cfg.sqp.tol = opts.sqpTol;
    cfg.optim.MaxIterations = opts.optimMaxIterations;
    cfg.optim.MaxFunctionEvaluations = opts.optimMaxFunctionEvaluations;
    cfg.save.enabled = opts.saveUnifiedResults;
    cfg.paths.resultDir = opts.outputDir;
    cfg.plot.enabled = false;
end

function resultsOut = run_unified_bayes_smm_task(cfg) %#ok<INUSD>
    run('bayesSMMUnifiedMMCompare.m');
    resultsOut = results;
end

function [errnorm, methodLabels, taskLabels] = collect_task_errors(taskResults, taskSpecs)
    numTasks = numel(taskResults);
    numMethods = numel(taskResults{1}.settings.methodNames);
    Ne = taskResults{1}.settings.cfg.experiment.Ne;
    errnorm = cell(1, numMethods);
    for jj = 1:numMethods
        errnorm{jj} = nan(Ne, numTasks);
    end

    for tt = 1:numTasks
        if numel(taskResults{tt}.settings.methodNames) ~= numMethods
            error('Task %s returned a different number of methods.', taskSpecs(tt).label);
        end
        if size(taskResults{tt}.errory, 1) ~= Ne
            error('Task %s returned a different number of experiments.', taskSpecs(tt).label);
        end
        for jj = 1:numMethods
            errnorm{jj}(:, tt) = taskResults{tt}.errory(:, jj);
        end
    end

    methodLabels = compact_method_labels(taskResults{1}.settings.methodNames);
    taskLabels = {taskSpecs.label};
end

function labels = compact_method_labels(methodNames)
    labels = methodNames;
    for jj = 1:numel(labels)
        switch labels{jj}
            case 'EmpBayes-MM'
                labels{jj} = 'EB-MM';
            case 'HierBayes-MM'
                labels{jj} = 'HB-MM';
            case 'N4SID+KF'
                labels{jj} = 'N4SID/KF';
        end
    end
end

function fig = plot_task_error_comparison(errnorm, methodLabels, taskLabels)
    fig = figure('Name', 'Unified Bayes SMM estimation error comparison');
    boxplotGroup(errnorm, 'PrimaryLabels', methodLabels, ...
        'SecondaryLabels', taskLabels, 'groupLabelType', 'Vertical', ...
        'interGroupSpace', 1, 'OutlierSize', 3);
    ylabel('Estimation error (RMSE)');
    grid on;
    if exist('fontsize', 'file')
        fontsize(9, 'points');
    else
        set(gca, 'FontSize', 9);
    end
    if exist('fontname', 'file')
        fontname('Times New Roman');
    else
        set(gca, 'FontName', 'Times New Roman');
    end
end
