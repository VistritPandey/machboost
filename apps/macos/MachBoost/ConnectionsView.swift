import SwiftUI

struct ConnectionsView: View {
    @Environment(AppState.self) private var appState
    @State private var endpoint = ""
    @State private var apiKey = ""
    @State private var requestedModel = ""
    @State private var requestNote = ""
    @State private var isConnecting = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                header
                modePicker
                if appState.inferenceMode == .local {
                    localStatus
                } else {
                    teamStatus
                }
            }
            .padding(28)
            .frame(maxWidth: 920, alignment: .leading)
        }
        .navigationTitle("Connections")
        .onAppear {
            if endpoint.isEmpty {
                endpoint = appState.teamHost?.endpoint.absoluteString ?? ""
            }
        }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: "point.3.connected.trianglepath.dotted")
                .font(.title2)
                .foregroundStyle(.green)
                .frame(width: 34, height: 34)
            VStack(alignment: .leading, spacing: 4) {
                Text("Inference source")
                    .font(.title2.weight(.semibold))
                Text(appState.inferenceLabel)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            connectionBadge
        }
    }

    private var modePicker: some View {
        Picker("Run models on", selection: modeBinding) {
            Label("This Mac", systemImage: "desktopcomputer")
                .tag(InferenceMode.local)
            Label("Host pool", systemImage: "server.rack")
                .tag(InferenceMode.team)
        }
        .pickerStyle(.segmented)
        .frame(maxWidth: 420)
    }

    private var localStatus: some View {
        VStack(alignment: .leading, spacing: 14) {
            sectionTitle("This Mac", systemImage: "apple.logo")
            HStack(spacing: 24) {
                metric("Ready models", value: "\(appState.catalog.filter(\.cached).count)")
                metric("Loaded", value: "\(appState.loadedModels.count)")
                metric("Endpoint", value: appState.configuration.endpoint.absoluteString)
            }
            Divider()
            Text("Repository tools and model inference both run on this Mac.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var teamStatus: some View {
        VStack(alignment: .leading, spacing: 22) {
            HStack {
                sectionTitle("Inference hosts", systemImage: "point.3.connected.trianglepath.dotted")
                Spacer()
                Toggle("Use this Mac when available", isOn: includeLocalBinding)
                    .toggleStyle(.switch)
            }

            if appState.teamHosts.isEmpty {
                Text("Add a host below. MachBoost routes each request to one compatible machine and spills over when another host is busy.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                VStack(spacing: 0) {
                    ForEach(appState.teamHosts) { host in
                        hostRow(host)
                        if host.id != appState.teamHosts.last?.id { Divider() }
                    }
                }
            }

            if !appState.hostDiscovery.hosts.isEmpty {
                Divider()
                nearbyHosts
            }
            Divider()
            connectForm
            if !appState.teamCatalog.isEmpty {
                Divider()
                hostModels
                Divider()
                modelRequest
            }
        }
    }

    private func hostRow(_ host: TeamHostProfile) -> some View {
        let snapshot = appState.teamHostSnapshots[host.id]
        return HStack(spacing: 14) {
            Image(systemName: snapshot?.isOnline == true ? "server.rack" : "server.rack")
                .foregroundStyle(snapshot?.isOnline == true ? Color.green : Color.secondary)
                .frame(width: 24)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 7) {
                    Text(host.hostName).font(.body.weight(.medium))
                    if appState.teamHost?.id == host.id {
                        Text("Preferred")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.green)
                    }
                }
                Text(host.endpoint.absoluteString)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            Spacer()
            if let snapshot {
                metric("Active", value: "\(snapshot.activeRequests)")
                metric("Queued", value: "\(snapshot.queuedRequests)")
                metric("Models", value: "\(snapshot.catalog.filter(\.cached).count)")
            }
            Button {
                appState.selectTeamHost(host)
            } label: {
                Image(systemName: "checkmark.circle")
            }
            .buttonStyle(.borderless)
            .disabled(snapshot?.isOnline != true || appState.teamHost?.id == host.id)
            .help("Prefer this host")
            Button(role: .destructive) {
                appState.removeTeamHost(host)
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.borderless)
            .help("Remove host")
        }
        .padding(.vertical, 10)
    }

    private var nearbyHosts: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Nearby MachBoost hosts").font(.headline)
            ForEach(appState.hostDiscovery.hosts) { host in
                HStack {
                    Label(host.name, systemImage: "bonjour")
                    Text(host.endpoint.absoluteString)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Use") {
                        endpoint = host.endpoint.absoluteString
                    }
                }
            }
        }
    }

    private var connectForm: some View {
        VStack(alignment: .leading, spacing: 16) {
            sectionTitle("Add host", systemImage: "link.badge.plus")
            Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 12) {
                GridRow {
                    Text("Endpoint")
                        .foregroundStyle(.secondary)
                    TextField("http://192.168.1.20:11435", text: $endpoint)
                        .textFieldStyle(.roundedBorder)
                        .frame(minWidth: 420)
                }
                GridRow {
                    Text("API key")
                        .foregroundStyle(.secondary)
                    SecureField("mbk_...", text: $apiKey)
                        .textFieldStyle(.roundedBorder)
                }
            }
            HStack {
                Button {
                    isConnecting = true
                    Task {
                        await appState.connectToTeamHost(endpoint: endpoint, token: apiKey)
                        isConnecting = false
                        if appState.teamIsConnected { apiKey = "" }
                    }
                } label: {
                    if isConnecting {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Label("Connect", systemImage: "link.badge.plus")
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
                .disabled(endpoint.isEmpty || apiKey.isEmpty || isConnecting)

                Text("The key is stored in Keychain.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var hostModels: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Available on host")
                .font(.headline)
            ForEach(appState.teamCatalog) { model in
                HStack(spacing: 10) {
                    Image(systemName: model.supportsVision ? "eye" : "text.bubble")
                        .foregroundStyle(.green)
                        .frame(width: 20)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(model.displayName)
                            .font(.body.weight(.medium))
                        Text(model.name)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                    Spacer()
                    if appState.teamLoadedModels.contains(where: {
                        $0.model == model.name || $0.model == model.repository
                    }) {
                        Label("Loaded", systemImage: "bolt.fill")
                            .font(.caption.weight(.medium))
                            .foregroundStyle(.green)
                    }
                }
                .padding(.vertical, 5)
            }
        }
    }

    private var modelRequest: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Request another model")
                .font(.headline)
            HStack(spacing: 10) {
                TextField("MLX repository or model alias", text: $requestedModel)
                    .textFieldStyle(.roundedBorder)
                TextField("Reason (optional)", text: $requestNote)
                    .textFieldStyle(.roundedBorder)
                Button {
                    let model = requestedModel
                    let note = requestNote
                    requestedModel = ""
                    requestNote = ""
                    Task { await appState.requestModelFromHost(model, note: note) }
                } label: {
                    Label("Request", systemImage: "paperplane")
                }
                .disabled(requestedModel.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
    }

    private var connectionBadge: some View {
        Label(
            appState.inferenceMode == .team ? "Team" : "Local",
            systemImage: appState.inferenceMode == .team ? "network" : "desktopcomputer"
        )
        .font(.caption.weight(.semibold))
        .foregroundStyle(appState.inferenceMode == .team ? Color.green : Color.secondary)
        .padding(.horizontal, 9)
        .frame(height: 26)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private var modeBinding: Binding<InferenceMode> {
        Binding(
            get: { appState.inferenceMode },
            set: { mode in
                switch mode {
                case .local:
                    appState.useLocalInference()
                case .team:
                    Task { await appState.useTeamInference() }
                }
            }
        )
    }

    private var includeLocalBinding: Binding<Bool> {
        Binding(
            get: { appState.includeLocalInHostPool },
            set: { appState.includeLocalInHostPool = $0 }
        )
    }

    private func sectionTitle(_ title: String, systemImage: String) -> some View {
        Label(title, systemImage: systemImage)
            .font(.headline)
    }

    private func metric(_ label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.callout.weight(.medium))
                .lineLimit(1)
                .textSelection(.enabled)
        }
    }
}
