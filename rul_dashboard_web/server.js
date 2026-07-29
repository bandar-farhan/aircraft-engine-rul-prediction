const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const express = require('express');
const dotenv = require('dotenv');

const app = express();
const PORT = Number(process.env.PORT || 3000);

const WEB_DIR = __dirname;
const PROJECT_DIR = path.resolve(WEB_DIR, '..');
const OUTPUT_DIR = path.join(PROJECT_DIR, 'outputs');
const DATASET_DIR = path.join(PROJECT_DIR, 'datasets');
const PYTHON_FILE = path.join(PROJECT_DIR, 'rul_prediction.py');
const DASHBOARD_GENERATOR = path.join(PROJECT_DIR, 'generate_dashboard_data.py');
const PUBLIC_DIR = path.join(WEB_DIR, 'public');

loadEnvFiles();

app.use(express.json({ limit: '1mb' }));
app.use(express.static(PUBLIC_DIR, { index: false }));
app.use('/logo.png', express.static(path.join(WEB_DIR, 'logo.png')));

app.use((req, res, next) => {
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
    next();
});

const RESULT_FILES = {
    finalKfoldResultsAllSubsets: 'final_kfold_results_all_subsets.csv',
    finalSplitResultsAllSubsets: 'final_split_results_all_subsets.csv',
    fd001Validation: 'fd001_validation_results.csv',
    fd001Test: 'fd001_test_results.csv',
    additionalResults: 'additional_results_fd002_fd004.csv',
    bestModels: 'best_models_fd002_fd004.csv',
    kfoldFd001: 'kfold_fd001.csv',
    kfoldSummaryFd001: 'kfold_summary_fd001.csv',
    kfoldFd002Fd004: 'kfold_fd002_fd004.csv',
    kfoldSummaryFd002Fd004: 'kfold_summary_fd002_fd004.csv',
    finalAllResults: 'final_all_results.csv',
    bestModelPerSubset: 'best_model_per_subset.csv'
};

const SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004'];
const ENGINE_FILE_GROUPS = {
    predictions: (subset) => `engine_predictions_${subset}.csv`,
    priority: (subset) => `maintenance_priority_${subset}.csv`,
    schedule: (subset) => `maintenance_schedule_${subset}.csv`
};

const chatHistories = new Map();
const MAX_HISTORY = 10;
let activeRun = null;
let runLogs = [];

function loadEnvFiles() {
    const envCandidates = [
        path.join(WEB_DIR, '.env'),
        path.join(PROJECT_DIR, '.env'),
        path.join(PROJECT_DIR, 'another project', 'src', 'other', '.env')
    ];

    for (const envPath of envCandidates) {
        if (fs.existsSync(envPath)) {
            dotenv.config({ path: envPath, override: false });
        }
    }
}

function fileExists(filePath) {
    return fs.existsSync(filePath);
}

function safeReadFile(filePath) {
    try {
        return fs.readFileSync(filePath, 'utf8');
    } catch {
        return null;
    }
}

function parseCsvLine(line) {
    const values = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const next = line[i + 1];

        if (char === '"' && inQuotes && next === '"') {
            current += '"';
            i++;
        } else if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            values.push(current);
            current = '';
        } else {
            current += char;
        }
    }

    values.push(current);
    return values;
}

function parseValue(value) {
    const trimmed = String(value ?? '').trim();
    if (trimmed === '') return '';
    const numeric = Number(trimmed);
    if (Number.isFinite(numeric) && /^-?\d+(\.\d+)?(e[-+]?\d+)?$/i.test(trimmed)) {
        return numeric;
    }
    return trimmed;
}

function parseCsv(content) {
    const normalized = content.replace(/^\uFEFF/, '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    const lines = normalized.split('\n').filter((line) => line.trim().length > 0);
    if (lines.length === 0) return [];

    const headers = parseCsvLine(lines[0]).map((header) => header.trim());
    return lines.slice(1).map((line) => {
        const values = parseCsvLine(line);
        const row = {};
        headers.forEach((header, index) => {
            row[header] = parseValue(values[index]);
        });
        return row;
    });
}

function readCsvIfExists(fileName) {
    const filePath = path.join(OUTPUT_DIR, fileName);
    if (!fs.existsSync(filePath)) {
        return { exists: false, rows: [], fileName };
    }

    const content = safeReadFile(filePath);
    if (content === null) {
        return { exists: true, rows: [], fileName, error: 'Unable to read file' };
    }

    try {
        return { exists: true, rows: parseCsv(content), fileName };
    } catch (error) {
        return { exists: true, rows: [], fileName, error: error.message };
    }
}

function getResultTables() {
    return Object.fromEntries(
        Object.entries(RESULT_FILES).map(([key, fileName]) => [key, readCsvIfExists(fileName)])
    );
}

function getEngineTables() {
    const data = {};

    for (const subset of SUBSETS) {
        data[subset] = {};
        for (const [group, fileBuilder] of Object.entries(ENGINE_FILE_GROUPS)) {
            data[subset][group] = readCsvIfExists(fileBuilder(subset));
        }
    }

    return data;
}

function getDatasetInfo() {
    return SUBSETS.map((subset) => {
        const train = datasetStats(`train_${subset}.txt`, `train_${subset}.csv`);
        const test = datasetStats(`test_${subset}.txt`, `test_${subset}.csv`);
        const rul = datasetStats(`RUL_${subset}.txt`, `RUL_${subset}.csv`);

        return {
            subset,
            trainRows: train.rows,
            trainEngines: train.engines,
            testRows: test.rows,
            testEngines: test.engines,
            rulRows: rul.rows,
            filesFound: train.exists && test.exists && rul.exists
        };
    });
}

function datasetStats(...names) {
    const filePath = names.map((name) => path.join(DATASET_DIR, name)).find((candidate) => fs.existsSync(candidate));
    if (!filePath) return { exists: false, rows: 0, engines: 0 };

    const content = safeReadFile(filePath);
    if (!content) return { exists: true, rows: 0, engines: 0 };

    const lines = content.split(/\r?\n/).filter((line) => line.trim().length > 0);
    const engines = new Set();

    for (const line of lines) {
        const first = line.trim().split(/\s+/)[0];
        if (/^\d+$/.test(first)) engines.add(first);
    }

    return { exists: true, rows: lines.length, engines: engines.size };
}

function summarizeResults(results, engines) {
    const availableResultFiles = Object.values(results).filter((table) => table.exists).length;
    const availableEngineFiles = Object.values(engines)
        .flatMap((subset) => Object.values(subset))
        .filter((table) => table.exists).length;

    const kfoldSummaryRows = [
        ...(results.finalKfoldResultsAllSubsets.rows || []),
        ...(results.kfoldSummaryFd001.rows || []),
        ...(results.kfoldSummaryFd002Fd004.rows || [])
    ];
    const finalRows = (results.finalKfoldResultsAllSubsets.rows || []).length
        ? results.finalKfoldResultsAllSubsets.rows
        : (results.finalAllResults.rows || []);
    const bestRows = results.bestModelPerSubset.rows || [];
    const comparisonRows = finalRows.length ? finalRows : kfoldSummaryRows;

    const bestMetricRow = comparisonRows
        .filter((row) => Number.isFinite(Number(row.RMSE_mean ?? row.RMSE)))
        .sort((a, b) => Number(a.RMSE_mean ?? a.RMSE) - Number(b.RMSE_mean ?? b.RMSE))[0] || null;

    const priorityRows = Object.values(engines).flatMap((subset) => subset.priority.rows || []);
    const criticalCount = priorityRows.filter((row) => row.maintenance_priority === 'critical').length;
    const highCount = priorityRows.filter((row) => row.maintenance_priority === 'high').length;
    const firstMaintenance = priorityRows
        .filter((row) => Number.isFinite(Number(row.ensemble_predicted_RUL)))
        .sort((a, b) => Number(a.ensemble_predicted_RUL) - Number(b.ensemble_predicted_RUL))[0] || null;

    return {
        availableResultFiles,
        availableEngineFiles,
        bestMetricRow,
        bestRows,
        priorityRows,
        criticalCount,
        highCount,
        firstMaintenance
    };
}

function bestRow(rows, metric) {
    return rows
        .filter((row) => Number.isFinite(Number(row[metric])))
        .sort((a, b) => Number(a[metric]) - Number(b[metric]))[0] || null;
}

function describeBestRow(label, row, metric) {
    if (!row) return null;
    const subset = row.Subset || 'FD001';
    const model = row.Model || 'unknown model';
    const value = row[metric];
    return `${label}: ${model} on ${subset} with ${metric}=${value}`;
}

function formatMetric(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return 'not available';
    return number.toFixed(2);
}

function bestAccuracyRowsFromPayload(payload) {
    const combinedKfold = payload.results.finalKfoldResultsAllSubsets?.rows || [];
    const fallbackKfold = [
        ...(payload.results.kfoldSummaryFd001?.rows || []),
        ...(payload.results.kfoldSummaryFd002Fd004?.rows || []),
        ...(payload.results.finalAllResults?.rows || [])
    ];
    const combinedSplit = payload.results.finalSplitResultsAllSubsets?.rows || [];
    const fallbackSplit = [
        ...(payload.results.fd001Validation?.rows || []).map((row) => ({ ...row, Subset: 'FD001', Split: 'Validation' })),
        ...(payload.results.fd001Test?.rows || []),
        ...(payload.results.additionalResults?.rows || [])
    ];

    return {
        kfoldRows: combinedKfold.length ? combinedKfold : fallbackKfold,
        splitRows: combinedSplit.length ? combinedSplit : fallbackSplit
    };
}

function buildDirectBestModelReply(message) {
    const normalized = message.toLowerCase();
    const asksBestModel = (
        normalized.includes('which model performed best')
        || normalized.includes('best model')
        || normalized.includes('best accuracy')
    );
    if (!asksBestModel) return null;

    const payload = buildDashboardPayload();
    const { kfoldRows, splitRows } = bestAccuracyRowsFromPayload(payload);
    const asksSplit = /\b(split|validation|test|standard)\b/.test(normalized);
    const asksMae = normalized.includes('mae') && !normalized.includes('rmse');

    if (asksSplit) {
        const metric = asksMae ? 'MAE' : 'RMSE';
        const row = bestRow(splitRows, metric);
        if (!row) return 'Split result data is not generated yet.';

        return `Best split result: ${row.Model} on ${row.Subset}${row.Split ? ` ${row.Split}` : ''} with MAE=${formatMetric(row.MAE)}, RMSE=${formatMetric(row.RMSE)}.`;
    }

    const metric = asksMae ? 'MAE_mean' : 'RMSE_mean';
    const row = bestRow(kfoldRows, metric);
    if (!row) return 'K-fold result data is not generated yet.';

    return `Best k-fold result: ${row.Model} on ${row.Subset} with MAE_mean=${formatMetric(row.MAE_mean)}, RMSE_mean=${formatMetric(row.RMSE_mean)}.`;
}

function getSortedPriorityRows(payload) {
    return (payload.summary.priorityRows || [])
        .filter((row) => Number.isFinite(Number(row.ensemble_predicted_RUL)))
        .slice()
        .sort((a, b) => Number(a.ensemble_predicted_RUL) - Number(b.ensemble_predicted_RUL));
}

function formatEngineRow(row) {
    const priorityLabels = {
        critical: 'Immediate Check',
        high: 'Upcoming Service',
        medium: 'Monitor',
        low: 'Healthy'
    };
    return `${row.subset} Engine ${row.engine_id}: ${formatMetric(row.ensemble_predicted_RUL)} cycles left, ${priorityLabels[row.maintenance_priority] || row.maintenance_priority}`;
}

function buildDirectDashboardReply(message) {
    const bestModelReply = buildDirectBestModelReply(message);
    if (bestModelReply) return bestModelReply;

    const normalized = message.toLowerCase();
    const asksEnginePriority = (
        normalized.includes('which engine')
        || normalized.includes('engine should')
        || normalized.includes('check first')
        || normalized.includes('needs attention')
        || normalized.includes('immediate')
        || normalized.includes('critical')
    );
    const asksPlan = normalized.includes('schedule') || normalized.includes('maintenance plan') || normalized.includes('plan');
    const asksHealthSummary = normalized.includes('engine health') || normalized.includes('today') || normalized.includes('summary');

    if (!asksEnginePriority && !asksPlan && !asksHealthSummary) return null;

    const payload = buildDashboardPayload();
    const rows = getSortedPriorityRows(payload);
    if (!rows.length) {
        return 'Engine health data is not ready yet. I cannot rank engines or create a maintenance plan without generated prediction files.';
    }

    if (asksPlan) {
        const critical = rows.filter((row) => row.maintenance_priority === 'critical').slice(0, 8);
        const high = rows.filter((row) => row.maintenance_priority === 'high').slice(0, 8);
        return [
            'Simple maintenance plan:',
            `Immediate Check: ${critical.length ? critical.map(formatEngineRow).join('; ') : 'none'}.`,
            `Upcoming Service: ${high.length ? high.map(formatEngineRow).join('; ') : 'none'}.`,
            'Use the dashboard table for the full ranked list.'
        ].join('\n');
    }

    if (asksHealthSummary && !asksEnginePriority) {
        const counts = rows.reduce((acc, row) => {
            acc[row.maintenance_priority] = (acc[row.maintenance_priority] || 0) + 1;
            return acc;
        }, {});
        const first = rows[0];
        return [
            `Engine health summary: Immediate Check ${counts.critical || 0}, Upcoming Service ${counts.high || 0}, Monitor ${counts.medium || 0}, Healthy ${counts.low || 0}.`,
            `First engine to check: ${formatEngineRow(first)}.`
        ].join('\n');
    }

    const firstRows = rows.slice(0, normalized.includes('all') || normalized.includes('list') ? 10 : 5);
    return `Engines to check first:\n${firstRows.map((row, index) => `${index + 1}. ${formatEngineRow(row)}. Action: ${row.suggested_action}`).join('\n')}`;
}

function buildDashboardPayload() {
    const results = getResultTables();
    const engines = getEngineTables();
    const datasets = getDatasetInfo();
    const summary = summarizeResults(results, engines);

    return {
        generatedAt: new Date().toISOString(),
        health: {
            outputDirExists: fs.existsSync(OUTPUT_DIR),
            datasetDirExists: fs.existsSync(DATASET_DIR),
            pythonFileExists: fs.existsSync(PYTHON_FILE),
            generatorExists: fs.existsSync(DASHBOARD_GENERATOR),
            openRouterConfigured: Boolean(process.env.OPENROUTER),
            activeRun: activeRun ? activeRun.type : null
        },
        datasets,
        results,
        engines,
        summary
    };
}

function buildAiContext(recentContext = '') {
    const payload = buildDashboardPayload();
    const summary = payload.summary;
    const standardRows = [
        ...((payload.results.finalSplitResultsAllSubsets?.rows || []).length
            ? payload.results.finalSplitResultsAllSubsets.rows
            : [
                ...(payload.results.fd001Validation?.rows || []).map((row) => ({ ...row, Subset: 'FD001', Split: 'Validation' })),
                ...(payload.results.fd001Test?.rows || []),
                ...(payload.results.additionalResults?.rows || [])
            ])
    ];
    const kfoldRows = [
        ...((payload.results.finalKfoldResultsAllSubsets?.rows || []).length
            ? payload.results.finalKfoldResultsAllSubsets.rows
            : [
                ...(payload.results.kfoldSummaryFd001?.rows || []),
                ...(payload.results.kfoldSummaryFd002Fd004?.rows || []),
                ...(payload.results.finalAllResults?.rows || [])
            ])
    ];
    const bestFactLines = [
        describeBestRow('Best standard MAE', bestRow(standardRows, 'MAE'), 'MAE'),
        describeBestRow('Best standard RMSE', bestRow(standardRows, 'RMSE'), 'RMSE'),
        describeBestRow('Best k-fold MAE_mean', bestRow(kfoldRows, 'MAE_mean'), 'MAE_mean'),
        describeBestRow('Best k-fold RMSE_mean', bestRow(kfoldRows, 'RMSE_mean'), 'RMSE_mean')
    ].filter(Boolean);

    let allPriorityRows = summary.priorityRows
        .slice()
        .sort((a, b) => Number(a.ensemble_predicted_RUL) - Number(b.ensemble_predicted_RUL));
        
    const topRows = allPriorityRows.slice(0, 10);

    const engineMatches = [...recentContext.matchAll(/engine\s*(\d+)|e(\d+)/gi)];
    for (const match of engineMatches) {
        const id = match[1] || match[2];
        const specificRows = allPriorityRows.filter(r => String(r.engine_id) === id);
        for (const row of specificRows) {
            if (!topRows.some(t => t.subset === row.subset && t.engine_id === row.engine_id)) {
                topRows.push(row);
            }
        }
    }

    const priorityLines = topRows.length
        ? topRows.map((row) => (
            `${row.subset} engine ${row.engine_id}: ensemble RUL ${row.ensemble_predicted_RUL}, priority ${row.maintenance_priority}, action ${row.suggested_action}`
        ))
        : ['No per-engine dashboard prediction CSV is available yet.'];

    const datasetSummary = payload.datasets
        .map((row) => `${row.subset}: ${row.filesFound ? 'ready' : 'missing files'}`)
        .join(', ');
    const resultFileCount = Object.values(payload.results).filter((table) => table.exists).length;

    return [
        'Project: Remaining Useful Life prediction for aircraft turbofan engines using NASA C-MAPSS FD001-FD004.',
        'Goal: support proactive maintenance by predicting RUL and ranking engines by maintenance urgency.',
        'Pipeline: load C-MAPSS data, calculate clipped RUL labels, split by engine, standard-scale features, create 30-cycle sequences, train BiLSTM, GRU+Attention, XGBoost, and average ensemble predictions.',
        'Metrics: MAE and RMSE. Lower values are better. K-Fold summaries use mean/std metrics.',
        'Maintenance priority thresholds: critical/Immediate Check <=20 cycles, high/Upcoming Service 21-50, medium/Monitor 51-90, low/Healthy >90. These are dashboard decision-support thresholds.',
        `Dataset status: ${datasetSummary}`,
        `Available result files: ${resultFileCount}`,
        `Best model facts: ${bestFactLines.length ? bestFactLines.join(' | ') : 'No best model facts available yet.'}`,
        summary.bestMetricRow
            ? `Dashboard displayed best accuracy result: ${summary.bestMetricRow.Model} on ${summary.bestMetricRow.Subset || 'FD001'} with MAE_mean=${summary.bestMetricRow.MAE_mean ?? 'not available'}, RMSE_mean=${summary.bestMetricRow.RMSE_mean ?? 'not available'}.`
            : 'No model result summary CSV is available yet.',
        `Engine priority counts: Immediate Check=${summary.criticalCount}, Upcoming Service=${summary.highCount}`,
        `Top maintenance rows: ${priorityLines.join(' | ')}`
    ].join('\n');
}

function isDeveloperInstructionProviderError(error) {
    const raw = error?.payload?.error?.metadata?.raw;
    return typeof raw === 'string' && raw.includes('Developer instruction is not enabled');
}

function isTransientOpenRouterError(error) {
    const status = error?.status;
    const payloadCode = error?.payload?.error?.code;
    const message = error?.payload?.error?.message || error?.message || '';
    return status === 504 || status === 503 || status === 429 || payloadCode === 429 || message.includes('The operation was aborted') || error?.name === 'AbortError';
}

function buildFallbackApiMessages(messages) {
    if (!Array.isArray(messages) || messages.length === 0) return messages;
    const [systemMessage, ...rest] = messages;
    if (!systemMessage || systemMessage.role !== 'system') return messages;
    if (!rest.length) return [{ role: 'user', content: systemMessage.content }];
    return [
        {
            ...rest[0],
            role: 'user',
            content: `${systemMessage.content}\n\nUser message:\n${rest[0].content}`
        },
        ...rest.slice(1)
    ];
}

function extractAssistantContent(payload) {
    const message = payload?.choices?.[0]?.message;
    const rawContent = message?.content || message?.reasoning || message?.reasoning_content;

    if (typeof rawContent === 'string') return rawContent.trim();
    if (Array.isArray(rawContent)) {
        return rawContent
            .map((part) => {
                if (typeof part === 'string') return part;
                if (typeof part?.text === 'string') return part.text;
                if (typeof part?.content === 'string') return part.content;
                return '';
            })
            .join('\n')
            .trim();
    }
    return '';
}

function summarizeOpenRouterPayload(payload) {
    const choice = payload?.choices?.[0];
    return {
        provider: payload?.provider || null,
        model: payload?.model || null,
        finishReason: choice?.finish_reason || choice?.native_finish_reason || null,
        hasContent: Boolean(extractAssistantContent(payload)),
        error: payload?.error
            ? {
                code: payload.error.code || null,
                message: payload.error.message || null,
                provider: payload.error.metadata?.provider_name || null
            }
            : null
    };
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

async function postOpenRouterRequest(requestConfig, requestOptions) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), requestOptions.timeout || 30000);

    try {
        const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
            method: 'POST',
            headers: requestOptions.headers,
            body: JSON.stringify(requestConfig),
            signal: controller.signal
        });
        const text = await response.text();
        let payload;
        try {
            payload = JSON.parse(text);
        } catch {
            payload = text;
        }

        if (!response.ok) {
            const error = new Error(`OpenRouter request failed with ${response.status}`);
            error.status = response.status;
            error.payload = payload;
            throw error;
        }

        return { data: payload };
    } finally {
        clearTimeout(timer);
    }
}

async function requestOpenRouterCompletion(apiMessages) {
    const requestConfig = {
        model: 'openrouter/free',
        messages: apiMessages,
        temperature: 0.7,
        max_tokens: 2000,
        top_p: 0.9
    };

    const requestOptions = {
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${process.env.OPENROUTER}`,
            'HTTP-Referer': 'http://localhost:3000',
            'X-Title': 'RUL Dashboard AI'
        },
        timeout: 30000
    };

    const executeRequest = async (messages, { retriesLeft = 3, allowSystemFallback = true } = {}) => {
        try {
            const response = await postOpenRouterRequest({ ...requestConfig, messages }, requestOptions);
            if (extractAssistantContent(response.data)) return response;

            if (retriesLeft > 0) {
                await sleep(2000);
                return executeRequest(messages, { retriesLeft: retriesLeft - 1, allowSystemFallback });
            }

            const invalidError = new Error('Invalid OpenRouter response');
            invalidError.status = 502;
            invalidError.payload = summarizeOpenRouterPayload(response.data);
            throw invalidError;
        } catch (error) {
            if (isTransientOpenRouterError(error) && retriesLeft > 0) {
                await sleep(2000);
                return executeRequest(messages, { retriesLeft: retriesLeft - 1, allowSystemFallback });
            }

            if (allowSystemFallback && isDeveloperInstructionProviderError(error)) {
                return executeRequest(buildFallbackApiMessages(messages), {
                    retriesLeft,
                    allowSystemFallback: false
                });
            }

            throw error;
        }
    };

    return executeRequest(apiMessages);
}

function buildSystemPrompt() {
    return `You are an AI assistant inside a Remaining Useful Life (RUL) dashboard for aircraft turbofan predictive maintenance.

Rules:
- Use only the dashboard context supplied by the server for numbers, rankings, schedules, and model results.
- Never invent missing MAE, RMSE, RUL, engine IDs, or maintenance schedules.
- If generated result files or engine prediction files are missing, say that clearly and tell the user which generation step is needed.
- Define RUL as remaining operating cycles in the C-MAPSS turbofan dataset. Do not describe it as hours unless the user provides hour-based data.
- Explain RUL labeling exactly: train RUL is max engine cycle minus current cycle, clipped at the configured cap; test RUL uses the provided final RUL file plus remaining observed cycles, then is clipped.
- Explain RUL, FD001-FD004, MAE, RMSE, BiLSTM, GRU+Attention, XGBoost, and ensemble results in clear language.
- When asked "Which model performed best?" without extra detail, answer ONLY from "Dashboard displayed best accuracy result". Do not mix MAE from one row with RMSE from another row.
- If the user asks for validation/test/standard/k-fold specifically, then distinguish those result types clearly.
- Use normal dashboard labels when discussing engine health: Immediate Check, Upcoming Service, Monitor, and Healthy.
- You may make maintenance-priority suggestions from the supplied thresholds, but present them as decision support, not certified aircraft safety instructions.
- Do not reveal API keys, .env values, hidden files, credentials, or internal secrets.
- Answer in the same language style as the user when practical.`;
}

function sanitizeMessage(message) {
    return String(message || '').trim().slice(0, 4000);
}

function getChatHistory(sessionId) {
    const key = String(sessionId || 'default').slice(0, 128);
    if (!chatHistories.has(key)) chatHistories.set(key, []);
    return { key, history: chatHistories.get(key) };
}

function appendRunLog(line) {
    runLogs.push(`[${new Date().toLocaleTimeString()}] ${line}`);
    if (runLogs.length > 300) runLogs = runLogs.slice(-300);
}

function startPythonRun(type, script, args = []) {
    if (activeRun) {
        return { started: false, error: `A ${activeRun.type} run is already active.` };
    }

    runLogs = [];
    appendRunLog(`Starting ${type}: python ${path.basename(script)} ${args.join(' ')}`);

    const child = spawn('python', [script, ...args], {
        cwd: PROJECT_DIR,
        shell: false,
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
    });

    activeRun = {
        type,
        startedAt: new Date().toISOString(),
        exitCode: null
    };

    child.stdout.on('data', (chunk) => appendRunLog(chunk.toString('utf8').trim()));
    child.stderr.on('data', (chunk) => appendRunLog(chunk.toString('utf8').trim()));
    child.on('close', (code) => {
        appendRunLog(`${type} finished with exit code ${code}`);
        activeRun = null;
    });

    return { started: true };
}

app.get('/', (req, res) => {
    res.sendFile(path.join(PUBLIC_DIR, 'index.html'));
});

app.get('/api/health', (req, res) => {
    const payload = buildDashboardPayload();
    res.json({
        ...payload.health,
        resultFiles: Object.fromEntries(
            Object.entries(RESULT_FILES).map(([key, fileName]) => [
                key,
                fs.existsSync(path.join(OUTPUT_DIR, fileName))
            ])
        ),
        engineFiles: Object.fromEntries(
            SUBSETS.map((subset) => [
                subset,
                Object.fromEntries(
                    Object.entries(ENGINE_FILE_GROUPS).map(([group, fileBuilder]) => [
                        group,
                        fs.existsSync(path.join(OUTPUT_DIR, fileBuilder(subset)))
                    ])
                )
            ])
        ),
        datasets: payload.datasets
    });
});

app.get('/api/results', (req, res) => {
    const payload = buildDashboardPayload();
    res.json({
        generatedAt: payload.generatedAt,
        datasets: payload.datasets,
        results: payload.results,
        summary: payload.summary
    });
});

app.get('/api/results/:name', (req, res) => {
    const entry = Object.entries(RESULT_FILES).find(([key]) => key === req.params.name);
    if (!entry) return res.status(404).json({ error: 'Unknown result table' });
    res.json(readCsvIfExists(entry[1]));
});

app.get('/api/engines', (req, res) => {
    res.json(getEngineTables());
});

app.get('/api/schedule', (req, res) => {
    const engines = getEngineTables();
    res.json(
        Object.fromEntries(
            SUBSETS.map((subset) => [subset, engines[subset].schedule])
        )
    );
});

app.post('/api/chat', async (req, res) => {
    const userMessage = sanitizeMessage(req.body?.message);
    if (!userMessage) return res.status(400).json({ error: 'Message is required.' });

    const { key, history } = getChatHistory(req.body?.sessionId);
    const directReply = buildDirectDashboardReply(userMessage);
    if (directReply) {
        history.push({ role: 'user', content: userMessage });
        history.push({ role: 'assistant', content: directReply });
        while (history.length > MAX_HISTORY * 2) history.shift();
        chatHistories.set(key, history);
        return res.json({ reply: directReply, model: 'dashboard-data' });
    }

    if (!process.env.OPENROUTER) {
        return res.status(500).json({ error: 'OPENROUTER key is not configured.' });
    }

    history.push({ role: 'user', content: userMessage });
    while (history.length > MAX_HISTORY * 2) history.shift();

    const recentContext = history.map(h => h.content).join(' ') + ' ' + userMessage;
    const apiMessages = [
        { role: 'system', content: `${buildSystemPrompt()}\n\nDashboard context:\n${buildAiContext(recentContext)}` },
        ...history
    ];

    try {
        const response = await requestOpenRouterCompletion(apiMessages);
        const content = extractAssistantContent(response.data);
        history.push({ role: 'assistant', content });
        while (history.length > MAX_HISTORY * 2) history.shift();
        chatHistories.set(key, history);
        res.json({ reply: content, model: response.data?.model || 'openrouter/free' });
    } catch (error) {
        if (history[history.length - 1]?.role === 'user') history.pop();
        const status = error.status || 500;
        const payload = typeof error.payload === 'object' ? summarizeOpenRouterPayload(error.payload) : null;
        console.error('AI chat error:', payload || error.message);
        res.status(status >= 400 && status < 600 ? status : 500).json({
            error: status === 401
                ? 'AI service configuration error.'
                : status === 429
                    ? 'AI service is busy or rate-limited. Try again later.'
                    : 'AI service is temporarily unavailable.'
        });
    }
});

app.post('/api/generate-dashboard-data', (req, res) => {
    const subset = SUBSETS.includes(req.body?.subset) ? req.body.subset : 'all';
    const args = subset === 'all' ? [] : ['--subset', subset];
    const result = startPythonRun('dashboard-data generation', DASHBOARD_GENERATOR, args);
    if (!result.started) return res.status(409).json(result);
    res.json({ success: true, message: 'Dashboard data generation started.' });
});

app.post('/api/run-full-model', (req, res) => {
    const result = startPythonRun('full RUL notebook workflow', PYTHON_FILE);
    if (!result.started) return res.status(409).json(result);
    res.json({ success: true, message: 'Full RUL workflow started. This can take a long time on CPU.' });
});

app.get('/api/run/logs', (req, res) => {
    res.json({ activeRun, logs: runLogs });
});

app.post('/api/shutdown', (req, res) => {
    if (activeRun) {
        return res.status(409).json({
            error: `Cannot stop the server while ${activeRun.type} is running.`
        });
    }

    res.json({ success: true, message: 'RUL Dashboard server is stopping.' });
    setTimeout(() => process.exit(0), 500);
});

app.listen(PORT, () => {
    console.log(`RUL Dashboard running at http://localhost:${PORT}`);
    console.log(`Project directory: ${PROJECT_DIR}`);
    console.log(`OpenRouter configured: ${process.env.OPENROUTER ? 'yes' : 'no'}`);
});
