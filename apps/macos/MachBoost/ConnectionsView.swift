import SwiftUI

struct ConnectionsView: View {
    @Environment(AppState.self) private var appState
    @State private var endpoint = ""
    @State private var apiKey = ""
    @State private var requestedModel = ""
    @State private var requestNote = ""
    @State private var isConnecting = false
    @State private var pendingNearbyHost: DiscoveredMachBoostHost?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                header
                currentDevice
                availableDevices
                connectedDevices
                advancedConnection
                if appState.inferenceMode == .team, !appState.teamCatalog.isEmpty {
                    remoteModels
                }
            }
            .padding(28)
            .frame(maxWidth: 920, alignment: .leading)
        }
        .navigationTitle("Connections")
        .sheet(item: $pendingNearbyHost) { host in
            NearbyHostConnectionSheet(
                host: host,
                isConnecting: isConnecting,
                onCancel: {
                    pendingNearbyHost = nil
                    apiKey = ""
                },
                onConnect: { token in
                    connect(endpoint: host.endpoint.absoluteString, token: token) {
                        pendingNearbyHost = nil
                    }
                }
            )
        }
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
                Text("Choose a device")
                    .font(.title2.weight(.semibold))
                Text("Run models here or connect to another MachBoost Mac on your network.")
                    .foregroundStyle(.secondary)
            }
            Spacer()
            connectionBadge
        }
    }

    private var currentDevice: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionTitle("This Mac", systemImage: "desktopcomputer")
            HStack(spacing: 14) {
                statusIndicator(active: appState.inferenceMode == .local)
                VStack(alignment: .leading, spacing: 3) {
                    Text(Host.current().localizedName ?? "This Mac")
                        .font(.body.weight(.medium))
                    Text("\(appState.catalog.filter(\.cached).count) models ready, \(appState.loadedModels.count) loaded")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if appState.inferenceMode == .local {
                    Label("In use", systemImage: "checkmark.circle.fill")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.green)
                } else {
                    Button("Use This Mac") {
                        appState.useLocalInference()
                    }
                    .buttonStyle(.bordered)
                }
            }
            .padding(.vertical, 8)
        }
    }

    @ViewBuilder
    private var availableDevices: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                sectionTitle("Available devices", systemImage: "wifi")
                Spacer()
                ProgressView()
                    .controlSize(.small)
                    .help("Looking for MachBoost devices")
            }

            if nearbyHosts.isEmpty {
                Text("No new MachBoost devices found yet. Make sure LAN access is enabled on the host, or connect by address below.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                VStack(spacing: 0) {
                    ForEach(nearbyHosts) { host in
                        HStack(spacing: 14) {
                            Image(systemName: "desktopcomputer.and.arrow.down")
                                .foregroundStyle(.green)
                                .frame(width: 24)
                            VStack(alignment: .leading, spacing: 3) {
                                Text(host.name)
                                    .font(.body.weight(.medium))
                                Text(host.endpoint.absoluteString)
                                    .font(.caption.monospaced())
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Connect") {
                                pendingNearbyHost = host
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.green)
                        }
                        .padding(.vertical, 10)
                        if host.id != nearbyHosts.last?.id { Divider() }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var connectedDevices: some View {
        if !appState.teamHosts.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    sectionTitle("Connected devices", systemImage: "link")
                    Spacer()
                    Toggle("Use this Mac as backup", isOn: includeLocalBinding)
                        .toggleStyle(.switch)
                        .help("Route work here when connected devices are busy")
                }
                VStack(spacing: 0) {
                    ForEach(appState.teamHosts) { host in
                        connectedHostRow(host)
                        if host.id != appState.teamHosts.last?.id { Divider() }
                    }
                }
            }
        }
    }

    private func connectedHostRow(_ host: TeamHostProfile) -> some View {
        let snapshot = appState.teamHostSnapshots[host.id]
        let selected = appState.inferenceMode == .team && appState.teamHost?.id == host.id
        return HStack(spacing: 14) {
            statusIndicator(active: snapshot?.isOnline == true)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 7) {
                    Text(host.hostName).font(.body.weight(.medium))
                    if selected {
                        Text("In use")
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
            Button(selected ? "Using" : "Use") {
                appState.selectTeamHost(host)
            }
            .buttonStyle(.bordered)
            .disabled(snapshot?.isOnline != true || selected)
            Button(role: .destructive) {
                appState.removeTeamHost(host)
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.borderless)
            .help("Forget device")
        }
        .padding(.vertical, 10)
    }

    private var advancedConnection: some View {
        DisclosureGroup("Connect by address") {
            VStack(alignment: .leading, spacing: 14) {
                Text("Use this when the device does not appear automatically.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("192.168.1.20:11435", text: $endpoint)
                    .textFieldStyle(.roundedBorder)
                SecureField("API key from the host", text: $apiKey)
                    .textFieldStyle(.roundedBorder)
                HStack {
                    Button {
                        connect(endpoint: endpoint, token: apiKey)
                    } label: {
                        if isConnecting {
                            ProgressView().controlSize(.small)
                        } else {
                            Label("Connect", systemImage: "link.badge.plus")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                    .disabled(
                        endpoint.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            || apiKey.isEmpty
                            || isConnecting
                    )
                    Text("The key is stored in Keychain.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.top, 12)
        }
        .font(.headline)
    }

    private var remoteModels: some View {
        VStack(alignment: .leading, spacing: 18) {
            Divider()
            VStack(alignment: .leading, spacing: 10) {
                Text("Models on \(appState.inferenceLabel)")
                    .font(.headline)
                ForEach(appState.teamCatalog) { model in
                    HStack(spacing: 10) {
                        Image(systemName: model.supportsVision ? "eye" : "text.bubble")
                            .foregroundStyle(.green)
                            .frame(width: 20)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(model.displayName).font(.body.weight(.medium))
                            Text(model.name)
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
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

            VStack(alignment: .leading, spacing: 10) {
                Text("Ask the host for another model")
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
    }

    private var nearbyHosts: [DiscoveredMachBoostHost] {
        let saved = Set(appState.teamHosts.map { $0.endpoint.absoluteString })
        return appState.hostDiscovery.hosts.filter { !saved.contains($0.endpoint.absoluteString) }
    }

    private func connect(endpoint: String, token: String, onSuccess: (() -> Void)? = nil) {
        isConnecting = true
        Task {
            await appState.connectToTeamHost(endpoint: endpoint, token: token)
            isConnecting = false
            if appState.teamIsConnected {
                apiKey = ""
                onSuccess?()
            }
        }
    }

    private var connectionBadge: some View {
        Label(
            appState.inferenceLabel,
            systemImage: appState.inferenceMode == .team ? "network" : "desktopcomputer"
        )
        .font(.caption.weight(.semibold))
        .foregroundStyle(appState.inferenceMode == .team ? Color.green : Color.secondary)
        .padding(.horizontal, 9)
        .frame(height: 26)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    private var includeLocalBinding: Binding<Bool> {
        Binding(
            get: { appState.includeLocalInHostPool },
            set: { appState.includeLocalInHostPool = $0 }
        )
    }

    private func statusIndicator(active: Bool) -> some View {
        Circle()
            .fill(active ? Color.green : Color.secondary.opacity(0.55))
            .frame(width: 9, height: 9)
            .accessibilityLabel(active ? "Available" : "Unavailable")
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
        }
    }
}

private struct NearbyHostConnectionSheet: View {
    let host: DiscoveredMachBoostHost
    let isConnecting: Bool
    let onCancel: () -> Void
    let onConnect: (String) -> Void
    @State private var apiKey = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 12) {
                Image(systemName: "desktopcomputer.and.arrow.down")
                    .font(.title2)
                    .foregroundStyle(.green)
                VStack(alignment: .leading, spacing: 3) {
                    Text("Connect to \(host.name)")
                        .font(.title3.weight(.semibold))
                    Text(host.endpoint.absoluteString)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            }
            SecureField("API key from the host", text: $apiKey)
                .textFieldStyle(.roundedBorder)
            Text("Find the key in MachBoost Settings on the host Mac.")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Spacer()
                Button("Cancel", action: onCancel)
                Button {
                    onConnect(apiKey)
                } label: {
                    if isConnecting {
                        ProgressView().controlSize(.small)
                    } else {
                        Text("Connect")
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
                .disabled(apiKey.isEmpty || isConnecting)
            }
        }
        .padding(24)
        .frame(width: 440)
    }
}
