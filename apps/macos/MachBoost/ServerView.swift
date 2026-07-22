import AppKit
import SwiftUI

struct ServerView: View {
    enum Mode: String, CaseIterable, Identifiable {
        case overview = "Overview"
        case developer = "Developer"
        var id: Self { self }
    }

    @Environment(AppState.self) private var appState
    @State private var mode: Mode = .overview
    @State private var draftConfiguration = ServerConfiguration()
    @State private var revealToken = false

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
                .frame(width: 220)
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
                    }
                }
                .padding(20)
                .frame(maxWidth: 960, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
        }
        .onAppear { draftConfiguration = appState.configuration }
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
            VStack(alignment: .leading, spacing: 8) {
                Text("Endpoint")
                    .font(.headline)
                CopyField(value: appState.configuration.endpoint.absoluteString)
            }

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

            SnippetView(title: "OpenAI Python", code: openAISnippet)
            SnippetView(title: "Ollama-compatible curl", code: ollamaSnippet)
        }
    }

    private var openAISnippet: String {
        let token = appState.configuration.lanEnabled ? (appState.apiToken ?? "YOUR_TOKEN") : "local"
        return """
        from openai import OpenAI

        client = OpenAI(
            base_url="\(appState.configuration.endpoint.absoluteString)/v1",
            api_key="\(token)",
        )
        response = client.chat.completions.create(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(response.choices[0].message.content)
        """
    }

    private var ollamaSnippet: String {
        let authorization = appState.configuration.lanEnabled
            ? "  -H 'Authorization: Bearer \(appState.apiToken ?? "YOUR_TOKEN")' \\" + "\n"
            : ""
        return """
        curl \(appState.configuration.endpoint.absoluteString)/api/chat \\
        \(authorization)  -H 'Content-Type: application/json' \\
          -d '{"model":"llama3.2:3b","messages":[{"role":"user","content":"Hello"}]}'
        """
    }

    private func formatBytes(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .memory)
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
