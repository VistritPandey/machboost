import AppKit
import SwiftUI

struct ServerView: View {
    enum Mode: String, CaseIterable, Identifiable {
        case overview = "Overview"
        case developer = "Developer"
        case team = "Team"
        case memory = "Memory & fallback"
        case logs = "Logs & evals"
        var id: Self { self }
    }

    @Environment(AppState.self) private var appState
    @State private var mode: Mode = .overview
    @State private var draftConfiguration = ServerConfiguration()
    @State private var revealToken = false
    @State private var newKeyName = ""
    @State private var allowedModels = ""
    @State private var keyConcurrency = 2
    @State private var keyRateLimit = 60
    @State private var traceMode = "metadata"
    @State private var retentionDays = 7
    @State private var keepTracesForever = false
    @State private var traceStorageMB = 256
    @State private var selectedTraceIDs: Set<String> = []
    @State private var judgeModel = ""
    @State private var loadModel = ""
    @State private var loadKeepAlive = "forever"
    @State private var compileWarmup = true
    @State private var memorySearch = ""
    @State private var providerName = ""
    @State private var providerURL = "https://"
    @State private var providerModels = ""
    @State private var providerAPIKey = ""
    @State private var providerBudget = ""

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Server")
                    .font(.title2.weight(.semibold))
                Picker("View", selection: $mode) {
                    ForEach(Mode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 560)
                Spacer()
                statusLabel
                Button {
                    Task {
                        if appState.serverIsRunning {
                            await appState.pauseServer()
                        } else {
                            await appState.resumeServer()
                        }
                    }
                } label: {
                    Label(
                        appState.serverIsRunning ? "Pause" : "Resume",
                        systemImage: appState.serverIsRunning ? "pause.fill" : "play.fill"
                    )
                }
                Button {
                    Task { await appState.refreshMetrics() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Refresh server")
            }
            .padding(18)

            Divider()

            ScrollView {
                Group {
                    switch mode {
                    case .overview:
                        overview
                    case .developer:
                        developer
                    case .team:
                        team
                    case .memory:
                        memoryAndFallback
                    case .logs:
                        logsAndEvaluations
                    }
                }
                .padding(20)
                .frame(maxWidth: 960, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
        }
        .onAppear {
            draftConfiguration = appState.configuration
            syncTeamSettings()
            selectLoadModelIfNeeded()
            Task { await appState.refreshMemoryAndProviders() }
        }
        .onChange(of: loadableModels.map(\.name)) { selectLoadModelIfNeeded() }
        .onChange(of: appState.teamStatus?.settings) {
            syncTeamSettings()
        }
        .onChange(of: appState.configuration) {
            draftConfiguration = appState.configuration
        }
        .task {
            while !Task.isCancelled {
                await appState.refreshMetrics()
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    private var statusLabel: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(appState.serverIsRunning ? Color.green : Color.red)
                .frame(width: 8, height: 8)
            Text(appState.serverIsRunning ? "Running" : "Stopped")
                .font(.caption.weight(.medium))
        }
    }

    private var overview: some View {
        VStack(alignment: .leading, spacing: 18) {
            metricsGrid
            modelLoader

            HStack {
                Text("Resident models")
                    .font(.headline)
                Spacer()
                Button {
                    Task { await appState.unloadAllModels() }
                } label: {
                    Label("Unload all", systemImage: "eject")
                }
                .disabled(appState.loadedModels.isEmpty)
            }

            if appState.loadedModels.isEmpty {
                ContentUnavailableView(
                    "No resident models",
                    systemImage: "memorychip",
                    description: Text("A model appears here after its first request or warm-up.")
                )
                .frame(maxWidth: .infinity, minHeight: 180)
            } else {
                ForEach(appState.loadedModels) { model in
                    residentModel(model)
                }
            }

            serverConfiguration

            DisclosureGroup("Recent daemon logs") {
                ScrollView(.horizontal) {
                    Text(appState.daemon.recentLogs.suffix(100).joined(separator: "\n"))
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                }
                .frame(minHeight: 120, maxHeight: 260)
                .background(Color(nsColor: .textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            }
        }
    }

    private var metricsGrid: some View {
        Grid(horizontalSpacing: 12, verticalSpacing: 12) {
            GridRow {
                MetricTile(
                    title: "Active",
                    value: "\(appState.metrics?.scheduler.activeRequests ?? 0)",
                    systemImage: "waveform.path"
                )
                MetricTile(
                    title: "Queued",
                    value: "\(appState.metrics?.scheduler.queuedRequests ?? 0)",
                    systemImage: "list.number"
                )
                MetricTile(
                    title: "Throughput",
                    value: "\((appState.metrics?.operations.generationTokensPerSecond ?? 0).formatted(.number.precision(.fractionLength(1)))) tok/s",
                    systemImage: "gauge.medium"
                )
            }
            GridRow {
                MetricTile(
                    title: "P50 latency",
                    value: formatLatency(appState.metrics?.operations.latencySeconds.p50 ?? 0),
                    systemImage: "timer"
                )
                MetricTile(
                    title: "P95 latency",
                    value: formatLatency(appState.metrics?.operations.latencySeconds.p95 ?? 0),
                    systemImage: "hourglass"
                )
                MetricTile(
                    title: "Peak memory",
                    value: formatBytes(appState.metrics?.process.peakResidentMemoryBytes ?? 0),
                    systemImage: "memorychip"
                )
            }
        }
    }

    private func residentModel(_ model: ModelInstance) -> some View {
        HStack(spacing: 12) {
            Image(systemName: model.capabilities.contains("vision") ? "eye.fill" : "text.bubble.fill")
                .foregroundStyle(model.capabilities.contains("vision") ? Color.indigo : Color.teal)
                .frame(width: 24)
            VStack(alignment: .leading, spacing: 3) {
                Text(model.model)
                    .font(.body.weight(.medium))
                    .lineLimit(1)
                Text("\(model.backend.uppercased()) · \(model.scheduler.replicas) replica(s) · \(model.requests) requests")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if model.scheduler.activeRequests > 0 || model.scheduler.queuedRequests > 0 {
                Text("\(model.scheduler.activeRequests) active · \(model.scheduler.queuedRequests) queued")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Button {
                Task { await appState.stop(model: model.model) }
            } label: {
                Image(systemName: "eject")
            }
            .help("Unload model")
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
    }

    private var serverConfiguration: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Serving configuration")
                .font(.headline)
            Toggle("Allow authenticated LAN access", isOn: $draftConfiguration.lanEnabled)
            Label(
                draftConfiguration.lanEnabled
                    ? "Remote clients will use \(draftConfiguration.advertisedEndpoint.absoluteString) with a bearer token."
                    : "Local only. Other computers cannot reach the loopback endpoint.",
                systemImage: draftConfiguration.lanEnabled ? "network" : "lock.fill"
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 10) {
                GridRow {
                    Text("Port")
                    TextField("Port", value: $draftConfiguration.port, format: .number)
                        .frame(width: 110)
                    Text("Text replicas")
                    Stepper(
                        value: $draftConfiguration.replicas,
                        in: 1...8
                    ) {
                        Text("\(draftConfiguration.replicas)")
                            .monospacedDigit()
                    }
                }
                GridRow {
                    Text("Queue limit")
                    TextField("Queue", value: $draftConfiguration.maxQueue, format: .number)
                        .frame(width: 110)
                    Text("Queue timeout")
                    TextField(
                        "Seconds",
                        value: $draftConfiguration.queueTimeout,
                        format: .number
                    )
                    .frame(width: 110)
                }
            }
            HStack {
                Button("Apply and restart") {
                    Task { await appState.applyConfiguration(draftConfiguration) }
                }
                .buttonStyle(.borderedProminent)
                .disabled(draftConfiguration == appState.configuration)
                Text("LAN mode binds all interfaces and requires the Keychain token.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.top, 6)
    }

    private var developer: some View {
        VStack(alignment: .leading, spacing: 18) {
            metricsGrid

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(appState.configuration.lanEnabled ? "LAN endpoint" : "Local endpoint")
                        .font(.headline)
                    Spacer()
                    Button(appState.configuration.lanEnabled ? "Disable LAN access" : "Enable authenticated LAN access") {
                        var configuration = appState.configuration
                        configuration.lanEnabled.toggle()
                        Task { await appState.applyConfiguration(configuration) }
                    }
                }
                CopyField(value: appState.configuration.advertisedEndpoint.absoluteString)
                Label(
                    appState.configuration.lanEnabled
                        ? "Other devices on this network can connect at this address with the API token."
                        : "127.0.0.1 is reachable only from this Mac.",
                    systemImage: appState.configuration.lanEnabled ? "checkmark.circle.fill" : "info.circle"
                )
                .font(.caption)
                .foregroundStyle(appState.configuration.lanEnabled ? Color.green : Color.secondary)
            }
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier("developer-endpoint-section")

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("API token")
                        .font(.headline)
                    Spacer()
                    Button {
                        revealToken.toggle()
                    } label: {
                        Image(systemName: revealToken ? "eye.slash" : "eye")
                    }
                    .help(revealToken ? "Hide token" : "Reveal token")
                    Button {
                        Task { await appState.rotateToken() }
                    } label: {
                        Image(systemName: "arrow.triangle.2.circlepath")
                    }
                    .help("Rotate token")
                }
                CopyField(
                    value: revealToken
                        ? (appState.apiToken ?? "Not generated")
                        : String(repeating: "•", count: 24),
                    copyValue: appState.apiToken
                )
                Text(appState.configuration.lanEnabled ? "Required for this server." : "Used after LAN access is enabled.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Loaded models")
                    .font(.headline)
                if appState.loadedModels.isEmpty {
                    Text("No resident models")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(appState.loadedModels) { model in
                        residentModel(model)
                    }
                }
            }

            modelLoader

            HStack(spacing: 8) {
                ProtocolBadge(title: "OpenAI Responses", systemImage: "sparkles")
                ProtocolBadge(title: "Anthropic Messages", systemImage: "hammer.fill")
                ProtocolBadge(title: "Ollama API", systemImage: "terminal.fill")
            }

            SnippetView(title: "OpenAI Responses", code: openAISnippet)
            SnippetView(title: "Codex custom provider", code: codexSnippet)
            SnippetView(title: "Claude Code gateway", code: claudeCodeSnippet)
            SnippetView(title: "Ollama-compatible curl", code: ollamaSnippet)
        }
    }

    private var team: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack(spacing: 12) {
                MetricTile(
                    title: "Active keys",
                    value: "\(appState.teamStatus?.keys ?? 0)",
                    systemImage: "key.fill"
                )
                MetricTile(
                    title: "Saved traces",
                    value: "\(appState.teamStatus?.traces ?? 0)",
                    systemImage: "waveform.path.ecg"
                )
                MetricTile(
                    title: "Evaluations",
                    value: "\(appState.teamStatus?.evaluations ?? 0)",
                    systemImage: "checkmark.seal.fill"
                )
            }

            VStack(alignment: .leading, spacing: 10) {
                Text("Create employee key")
                    .font(.headline)
                TextField("Name", text: $newKeyName)
                TextField("Allowed models, comma-separated (blank allows all)", text: $allowedModels)
                HStack(spacing: 18) {
                    Stepper("Concurrent \(keyConcurrency)", value: $keyConcurrency, in: 1...16)
                    Stepper("Requests/min \(keyRateLimit)", value: $keyRateLimit, in: 1...600, step: 10)
                    Spacer()
                    Button {
                        Task {
                            await appState.createTeamKey(
                                name: newKeyName,
                                allowedModels: allowedModels
                                    .split(separator: ",")
                                    .map { $0.trimmingCharacters(in: .whitespaces) }
                                    .filter { !$0.isEmpty },
                                maxConcurrent: keyConcurrency,
                                requestsPerMinute: keyRateLimit
                            )
                            newKeyName = ""
                        }
                    } label: {
                        Label("Create key", systemImage: "key.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(newKeyName.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier("team-key-section")

            if let token = appState.lastCreatedTeamToken {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("New key")
                            .font(.headline)
                        Spacer()
                        Button {
                            appState.clearCreatedTeamToken()
                        } label: {
                            Image(systemName: "xmark")
                        }
                        .help("Dismiss key")
                    }
                    CopyField(value: token)
                    Text("Shown once. Store it in the employee's client configuration.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(12)
                .background(Color.green.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            }

            VStack(alignment: .leading, spacing: 10) {
                Text("Employee access")
                    .font(.headline)
                if appState.teamKeys.isEmpty {
                    ContentUnavailableView("No employee keys", systemImage: "person.badge.key")
                        .frame(maxWidth: .infinity, minHeight: 120)
                } else {
                    ForEach(appState.teamKeys) { key in
                        HStack(spacing: 12) {
                            Image(systemName: key.enabled == false ? "key.slash" : "key.fill")
                                .foregroundStyle(key.enabled == false ? .secondary : Color.green)
                                .frame(width: 24)
                            VStack(alignment: .leading, spacing: 3) {
                                Text(key.name)
                                    .font(.body.weight(.medium))
                                Text("\(key.maxConcurrent) concurrent · \(key.requestsPerMinute)/min · \(key.allowedModels.isEmpty ? "all models" : "\(key.allowedModels.count) models")")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button(role: .destructive) {
                                Task { await appState.revokeTeamKey(id: key.id) }
                            } label: {
                                Image(systemName: "trash")
                            }
                            .help("Revoke key")
                            .disabled(key.enabled == false)
                        }
                        .padding(.vertical, 6)
                        Divider()
                    }
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                Text("Coding fleet readiness")
                    .font(.headline)
                HStack(spacing: 16) {
                    ReadinessItem(
                        title: "LAN endpoint",
                        ready: appState.configuration.lanEnabled
                    )
                    ReadinessItem(
                        title: "Native model",
                        ready: !appState.loadedModels.filter { $0.backend != "ollama-mlx" }.isEmpty
                    )
                    ReadinessItem(
                        title: "Employee keys",
                        ready: !appState.teamKeys.isEmpty
                    )
                    Spacer()
                }
            }

            SnippetView(title: "Team environment", code: teamEnvironmentSnippet)
            SnippetView(title: "Codex config.toml", code: codexSnippet)
            SnippetView(title: "Claude Code", code: claudeCodeSnippet)
        }
    }

    private var memoryAndFallback: some View {
        VStack(alignment: .leading, spacing: 22) {
            HStack {
                Text("Reuse & savings")
                    .font(.headline)
                Spacer()
                Button {
                    Task { await appState.refreshMemoryAndProviders() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Refresh memory and providers")
            }

            Grid(horizontalSpacing: 12, verticalSpacing: 12) {
                GridRow {
                    MetricTile(
                        title: "Memories",
                        value: "\(appState.memories.count)",
                        systemImage: "brain.head.profile"
                    )
                    MetricTile(
                        title: "Exact hits",
                        value: metricValue("exact_cache_hits"),
                        systemImage: "bolt.fill"
                    )
                    MetricTile(
                        title: "Prompt tokens avoided",
                        value: metricValue("avoided_prompt_tokens"),
                        systemImage: "text.badge.checkmark"
                    )
                    MetricTile(
                        title: "Estimated cost avoided",
                        value: avoidedCost,
                        systemImage: "dollarsign.circle"
                    )
                }
            }

            Divider()

            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Team memory ledger")
                        .font(.headline)
                    Spacer()
                    TextField("Search memories", text: $memorySearch)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 280)
                }
                if filteredMemories.isEmpty {
                    ContentUnavailableView(
                        "No matching memory",
                        systemImage: "tray",
                        description: Text("Workspace chats create private memories; administrators can publish validated team entries through the API.")
                    )
                    .frame(maxWidth: .infinity, minHeight: 150)
                } else {
                    ForEach(filteredMemories.prefix(100)) { memory in
                        HStack(alignment: .top, spacing: 12) {
                            Image(systemName: memory.scope == "team" ? "person.3.fill" : "lock.fill")
                                .foregroundStyle(memory.scope == "team" ? Color.green : .secondary)
                                .frame(width: 24)
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text(memory.title)
                                        .font(.body.weight(.medium))
                                    Text(memory.kind.uppercased())
                                        .font(.caption2.weight(.semibold))
                                        .foregroundStyle(.secondary)
                                }
                                Text(memory.content)
                                    .font(.callout)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(3)
                                Text("\(memory.scope) · confidence \(memory.confidence.formatted(.number.precision(.fractionLength(2))))")
                                    .font(.caption)
                                    .foregroundStyle(.tertiary)
                            }
                            Spacer()
                            Button(role: .destructive) {
                                Task { await appState.deleteMemory(id: memory.id) }
                            } label: {
                                Image(systemName: "trash")
                            }
                            .help("Delete memory")
                        }
                        Divider()
                    }
                }
            }

            Divider()

            VStack(alignment: .leading, spacing: 12) {
                Text("External fallback")
                    .font(.headline)
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 10) {
                    GridRow {
                        Text("Name")
                        TextField("Production inference", text: $providerName)
                    }
                    GridRow {
                        Text("Base URL")
                        TextField("https://api.example.com", text: $providerURL)
                    }
                    GridRow {
                        Text("Models")
                        TextField("model-a, model-b or *", text: $providerModels)
                    }
                    GridRow {
                        Text("API key")
                        SecureField("Stored in Keychain", text: $providerAPIKey)
                    }
                    GridRow {
                        Text("Monthly budget")
                        TextField("Optional USD", text: $providerBudget)
                    }
                }
                HStack {
                    Text("Remote endpoints must use HTTPS. Keys are stored in macOS Keychain and are not written to the team database or logs.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button {
                        let models = providerModels
                            .split(separator: ",")
                            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                            .filter { !$0.isEmpty }
                        Task {
                            await appState.configureProvider(
                                id: nil,
                                name: providerName,
                                baseURL: providerURL,
                                models: models,
                                apiKey: providerAPIKey,
                                monthlyBudgetUSD: Double(providerBudget)
                            )
                            providerAPIKey = ""
                        }
                    } label: {
                        Label("Add provider", systemImage: "plus")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(
                        providerName.trimmingCharacters(in: .whitespaces).isEmpty
                            || providerURL.trimmingCharacters(in: .whitespaces).isEmpty
                            || providerModels.trimmingCharacters(in: .whitespaces).isEmpty
                    )
                }

                ForEach(appState.providers) { provider in
                    HStack(spacing: 12) {
                        Image(systemName: provider.hasSecret ? "checkmark.shield.fill" : "exclamationmark.triangle.fill")
                            .foregroundStyle(provider.hasSecret ? Color.green : Color.orange)
                            .frame(width: 24)
                        VStack(alignment: .leading, spacing: 3) {
                            Text(provider.name)
                                .font(.body.weight(.medium))
                            Text("\(provider.baseURL) · \(provider.models.joined(separator: ", "))")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                            Text("Spent $\(provider.spentThisMonthUSD.formatted(.number.precision(.fractionLength(4)))) this month")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                        Spacer()
                        Button(role: .destructive) {
                            Task { await appState.deleteProvider(id: provider.id) }
                        } label: {
                            Image(systemName: "trash")
                        }
                        .help("Delete provider")
                    }
                    Divider()
                }
            }
        }
    }

    private var logsAndEvaluations: some View {
        VStack(alignment: .leading, spacing: 20) {
            VStack(alignment: .leading, spacing: 12) {
                Text("Trace policy")
                    .font(.headline)
                Picker("Content", selection: $traceMode) {
                    Text("Off").tag("off")
                    Text("Metadata").tag("metadata")
                    Text("Redacted").tag("redacted")
                    Text("Full").tag("full")
                }
                .pickerStyle(.segmented)
                Toggle("Keep until storage limit", isOn: $keepTracesForever)
                HStack(spacing: 18) {
                    Stepper("Retention \(retentionDays) days", value: $retentionDays, in: 1...365)
                        .disabled(keepTracesForever)
                    Stepper("Storage \(traceStorageMB) MB", value: $traceStorageMB, in: 32...4096, step: 32)
                    Spacer()
                    Button("Save policy") {
                        Task {
                            await appState.updateTeamSettings(
                                traceMode: traceMode,
                                retentionDays: keepTracesForever ? nil : retentionDays,
                                maxStorageBytes: Int64(traceStorageMB) * 1024 * 1024
                            )
                        }
                    }
                    .buttonStyle(.borderedProminent)
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Request traces")
                        .font(.headline)
                    Spacer()
                    Button {
                        Task { await appState.refreshTeam() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .help("Refresh traces")
                }
                if appState.traces.isEmpty {
                    ContentUnavailableView("No retained traces", systemImage: "waveform.path.ecg")
                        .frame(maxWidth: .infinity, minHeight: 120)
                } else {
                    ForEach(appState.traces.prefix(50)) { trace in
                        Toggle(isOn: traceSelection(trace.id)) {
                            HStack {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(trace.model)
                                        .font(.body.weight(.medium))
                                        .lineLimit(1)
                                    Text("\(trace.principal.name) · \(trace.endpoint) · \(trace.completionTokens) tokens")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                Text(formatLatency(trace.durationSeconds))
                                    .font(.caption.monospacedDigit())
                                Image(systemName: trace.status == "completed" ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                                    .foregroundStyle(trace.status == "completed" ? Color.green : Color.red)
                            }
                        }
                        .toggleStyle(.checkbox)
                        Divider()
                    }
                }
            }

            HStack {
                TextField("Optional local judge model", text: $judgeModel)
                Button("Evaluate selected") {
                    Task {
                        await appState.evaluateTraces(
                            ids: Array(selectedTraceIDs),
                            model: judgeModel.trimmingCharacters(in: .whitespaces).isEmpty
                                ? nil
                                : judgeModel
                        )
                        selectedTraceIDs.removeAll()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(selectedTraceIDs.isEmpty)
            }

            if let latest = appState.evaluations.first {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Latest evaluation")
                        .font(.headline)
                    HStack(spacing: 20) {
                        LabeledContent("Evaluator", value: latest.evaluator)
                        LabeledContent("Completion", value: latest.summary.completionRate.formatted(.percent.precision(.fractionLength(0))))
                        LabeledContent("P50", value: formatLatency(latest.summary.latencySeconds.p50))
                        LabeledContent("Throughput", value: "\(latest.summary.generationTokensPerSecond.formatted(.number.precision(.fractionLength(1)))) tok/s")
                    }
                }
            }
        }
    }

    private var teamEnvironmentSnippet: String {
        return """
        export OPENAI_BASE_URL="\(teamEndpoint)/v1"
        export OPENAI_API_KEY="YOUR_MACHBOOST_KEY"
        export OLLAMA_HOST="\(teamEndpoint)"
        export ANTHROPIC_BASE_URL="\(teamEndpoint)"
        export ANTHROPIC_AUTH_TOKEN="YOUR_MACHBOOST_KEY"
        export ANTHROPIC_MODEL="\(preferredServerModel)"
        """
    }

    private var teamEndpoint: String {
        appState.configuration.lanEnabled
            ? appState.configuration.advertisedEndpoint.absoluteString
            : "http://YOUR_MACHBOOST_MAC_IP:\(appState.configuration.port)"
    }

    private var filteredMemories: [MemorySummary] {
        let query = memorySearch.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return appState.memories }
        return appState.memories.filter {
            $0.title.localizedCaseInsensitiveContains(query)
                || $0.content.localizedCaseInsensitiveContains(query)
                || $0.kind.localizedCaseInsensitiveContains(query)
        }
    }

    private func metricValue(_ key: String) -> String {
        (appState.cacheMetrics?.totals[key] ?? 0).formatted()
    }

    private var avoidedCost: String {
        let microdollars = appState.cacheMetrics?.totals["avoided_cost_microusd"] ?? 0
        return "$\((Double(microdollars) / 1_000_000).formatted(.number.precision(.fractionLength(2))))"
    }

    private var loadableModels: [CatalogModel] {
        appState.catalog
            .filter {
                $0.cached
                    && $0.support == "ready"
                    && ($0.backend.hasPrefix("mlx") || $0.backend == "dflash")
            }
            .sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }
    }

    private var selectedLoadModel: CatalogModel? {
        loadableModels.first { $0.name == loadModel }
    }

    private var selectedModelIsLoaded: Bool {
        guard let selectedLoadModel else { return false }
        return appState.loadedModels.contains {
            $0.model == selectedLoadModel.name || $0.model == selectedLoadModel.repository
        }
    }

    private var modelLoader: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Load a resident model")
                .font(.headline)
            if loadableModels.isEmpty {
                Label("Download a compatible model before loading it.", systemImage: "arrow.down.circle")
                    .foregroundStyle(.secondary)
            } else {
                HStack(spacing: 12) {
                    Menu {
                        ForEach(loadableModels) { model in
                            Button {
                                loadModel = model.name
                            } label: {
                                Label(
                                    model.displayName,
                                    systemImage: model.capabilities.contains("vision")
                                        ? "eye.fill"
                                        : "text.bubble.fill"
                                )
                            }
                        }
                    } label: {
                        HStack(spacing: 10) {
                            Image(
                                systemName: selectedLoadModel?.capabilities.contains("vision") == true
                                    ? "eye.fill"
                                    : "memorychip.fill"
                            )
                            .foregroundStyle(.green)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(selectedLoadModel?.displayName ?? "Select model")
                                    .font(.body.weight(.medium))
                                    .lineLimit(1)
                                if let model = selectedLoadModel {
                                    Text(modelLoaderSubtitle(model))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                            }
                            Spacer(minLength: 8)
                            Image(systemName: "chevron.up.chevron.down")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 10)
                        .frame(width: 340, height: 48)
                        .background(Color(nsColor: .textBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
                        }
                    }
                    .accessibilityIdentifier("resident-model-picker")

                    Picker("Keep loaded", selection: $loadKeepAlive) {
                        Text("15 minutes").tag("15m")
                        Text("1 hour").tag("1h")
                        Text("Forever").tag("forever")
                    }
                    .frame(width: 190)

                    Toggle("Compile warm-up", isOn: $compileWarmup)
                        .toggleStyle(.checkbox)

                    Spacer()

                    Button {
                        Task {
                            await appState.load(
                                model: loadModel,
                                keepAlive: loadKeepAlive,
                                warmup: compileWarmup
                            )
                        }
                    } label: {
                        if appState.loadingModels.contains(loadModel) {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Label(
                                selectedModelIsLoaded ? "Warm again" : "Load",
                                systemImage: selectedModelIsLoaded ? "arrow.clockwise" : "play.fill"
                            )
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .accessibilityIdentifier("load-resident-model")
                    .disabled(loadModel.isEmpty || appState.loadingModels.contains(loadModel))
                }
                Text("Loading keeps the model in unified memory before the first client request. Warm-up compiles its initial generation path.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if let result = appState.lastModelLoad, result.model == loadModel {
                    Label(
                        "Resident model ready",
                        systemImage: "checkmark.circle.fill"
                    )
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.green)
                }
            }
        }
    }

    private func selectLoadModelIfNeeded() {
        if !loadableModels.contains(where: { $0.name == loadModel }) {
            loadModel = loadableModels.first?.name ?? ""
        }
    }

    private func modelLoaderSubtitle(_ model: CatalogModel) -> String {
        let size = model.diskSizeGB ?? model.downloadSizeGB
        let sizeText = size.map { "\($0.formatted(.number.precision(.fractionLength(1)))) GB" }
        return ([model.backend.uppercased(), sizeText].compactMap { $0 }).joined(separator: " · ")
    }

    private func traceSelection(_ id: String) -> Binding<Bool> {
        Binding(
            get: { selectedTraceIDs.contains(id) },
            set: { selected in
                if selected {
                    selectedTraceIDs.insert(id)
                } else {
                    selectedTraceIDs.remove(id)
                }
            }
        )
    }

    private func syncTeamSettings() {
        guard let settings = appState.teamStatus?.settings else { return }
        traceMode = settings.traceMode
        keepTracesForever = settings.retentionDays == nil
        retentionDays = settings.retentionDays ?? 7
        traceStorageMB = max(32, Int(settings.maxStorageBytes / 1024 / 1024))
    }

    private var openAISnippet: String {
        let token = appState.configuration.lanEnabled ? (appState.apiToken ?? "YOUR_TOKEN") : "local"
        return """
        from openai import OpenAI

        client = OpenAI(
            base_url="\(appState.configuration.advertisedEndpoint.absoluteString)/v1",
            api_key="\(token)",
        )
        response = client.responses.create(
            model="\(preferredServerModel)",
            input="Inspect this repository and propose the smallest fix.",
        )
        print(response.output_text)
        """
    }

    private var codexSnippet: String {
        """
        model = "\(preferredServerModel)"
        model_provider = "machboost"

        [model_providers.machboost]
        name = "MachBoost"
        base_url = "\(teamEndpoint)/v1"
        env_key = "MACHBOOST_API_KEY"
        wire_api = "responses"

        # In the employee shell:
        # export MACHBOOST_API_KEY="YOUR_MACHBOOST_KEY"
        """
    }

    private var claudeCodeSnippet: String {
        """
        export ANTHROPIC_BASE_URL="\(teamEndpoint)"
        export ANTHROPIC_AUTH_TOKEN="YOUR_MACHBOOST_KEY"
        export ANTHROPIC_MODEL="\(preferredServerModel)"
        claude
        """
    }

    private var preferredServerModel: String {
        if let loaded = appState.loadedModels.first(where: { $0.backend != "ollama-mlx" }) {
            return loaded.model
        }
        if let selectedLoadModel {
            return selectedLoadModel.name
        }
        return "qwen2.5-coder:7b"
    }

    private var ollamaSnippet: String {
        let authorization = appState.configuration.lanEnabled
            ? "  -H 'Authorization: Bearer \(appState.apiToken ?? "YOUR_TOKEN")' \\" + "\n"
            : ""
        return """
        curl \(appState.configuration.advertisedEndpoint.absoluteString)/api/chat \\
        \(authorization)  -H 'Content-Type: application/json' \\
          -d '{"model":"\(preferredServerModel)","messages":[{"role":"user","content":"Hello"}]}'
        """
    }

    private func formatBytes(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .memory)
    }

    private func formatLatency(_ seconds: Double) -> String {
        if seconds < 1 {
            return "\((seconds * 1_000).formatted(.number.precision(.fractionLength(0)))) ms"
        }
        return "\(seconds.formatted(.number.precision(.fractionLength(2)))) s"
    }
}

private struct MetricTile: View {
    let title: String
    let value: String
    let systemImage: String

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: systemImage)
                .foregroundStyle(.teal)
                .frame(width: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(value)
                    .font(.headline.monospacedDigit())
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
                Text(title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .frame(minWidth: 150, minHeight: 64)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
    }
}

private struct ProtocolBadge: View {
    let title: String
    let systemImage: String

    var body: some View {
        Label(title, systemImage: systemImage)
            .font(.caption.weight(.medium))
            .foregroundStyle(.green)
            .padding(.horizontal, 9)
            .frame(height: 28)
            .background(Color.green.opacity(0.1))
            .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
    }
}

private struct ReadinessItem: View {
    let title: String
    let ready: Bool

    var body: some View {
        Label(title, systemImage: ready ? "checkmark.circle.fill" : "circle")
            .font(.callout.weight(.medium))
            .foregroundStyle(ready ? Color.green : Color.secondary)
    }
}

private struct CopyField: View {
    let value: String
    var copyValue: String? = nil

    var body: some View {
        HStack {
            Text(value)
                .font(.system(.body, design: .monospaced))
                .lineLimit(1)
                .textSelection(.enabled)
            Spacer()
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(copyValue ?? value, forType: .string)
            } label: {
                Image(systemName: "doc.on.doc")
            }
            .help("Copy")
        }
        .padding(.horizontal, 10)
        .frame(height: 36)
        .background(Color(nsColor: .textBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
        }
    }
}

private struct SnippetView: View {
    let title: String
    let code: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title)
                    .font(.headline)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(code, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .help("Copy snippet")
            }
            ScrollView(.horizontal) {
                Text(code)
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(12)
            }
            .background(Color(nsColor: .textBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
            }
        }
    }
}
