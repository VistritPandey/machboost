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
            VStack(alignment: .leading, spacing: 24) {
                header
                inferencePool
                availableDevices
                advancedConnection
                if appState.inferenceMode == .team, !appState.teamCatalog.isEmpty {
                    remoteModels
                }
            }
            .padding(24)
            .frame(maxWidth: 1120, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
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
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 14) {
                connectionHeader
                Spacer(minLength: 16)
                connectionBadge
            }
            VStack(alignment: .leading, spacing: 12) {
                connectionHeader
                connectionBadge
            }
        }
    }

    private var connectionHeader: some View {
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: "point.3.connected.trianglepath.dotted")
                .font(.title2)
                .foregroundStyle(.green)
                .frame(width: 34, height: 34)
            VStack(alignment: .leading, spacing: 4) {
                Text("Inference devices")
                    .font(.title2.weight(.semibold))
                Text("Use this Mac, connect to another Mac, or let the pool route around busy hosts.")
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var inferencePool: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                sectionTitle("Your inference pool", systemImage: "point.3.filled.connected.trianglepath.dotted")
                Spacer()
                Toggle("Use this Mac as backup", isOn: includeLocalBinding)
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .help("Route work here when connected devices are busy")
            }
            Text("Every request is sent to the ready device with the lowest expected completion time. Live latency, queues, replicas, and model residency are refreshed automatically.")
                .font(.callout)
                .foregroundStyle(.secondary)

            if let delay = appState.lastRouteExpectedDelay {
                Label(
                    "Last request used \(appState.inferenceLabel) · estimated \(delay.formatted(.number.precision(.fractionLength(2))))s",
                    systemImage: "arrow.triangle.branch"
                )
                .font(.caption.weight(.medium))
                .foregroundStyle(.green)
            }

            ScrollView(.horizontal) {
                HStack(alignment: .center, spacing: 10) {
                    localHostNode
                    ForEach(appState.teamHosts) { host in
                        Image(systemName: "arrow.left.arrow.right")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .accessibilityHidden(true)
                        remoteHostNode(host)
                    }
                }
                .padding(.vertical, 2)
            }
            .scrollIndicators(.hidden)
        }
    }

    private var localHostNode: some View {
        let selected = appState.inferenceMode == .local || appState.lastRouteWasLocal
        return hostNode(
            name: Host.current().localizedName ?? "This Mac",
            subtitle: "This Mac",
            online: true,
            selected: selected,
            active: appState.metrics?.scheduler.activeRequests ?? 0,
            queued: appState.metrics?.scheduler.queuedRequests ?? 0,
            models: appState.catalog.filter(\.cached).count,
            loaded: appState.loadedModels.count,
            latency: 0,
            useAction: appState.inferenceMode == .local ? nil : { appState.useLocalInference() },
            removeAction: nil
        )
    }

    private func remoteHostNode(_ host: TeamHostProfile) -> some View {
        let snapshot = appState.teamHostSnapshots[host.id]
        let selected = appState.inferenceMode == .team && appState.wasLastRouted(to: host)
        let useAction: (() -> Void)? = snapshot?.isOnline == true && appState.inferenceMode == .local
            ? { appState.selectTeamHost(host) }
            : nil
        return hostNode(
            name: host.hostName,
            subtitle: snapshot?.isOnline == true ? host.endpoint.host ?? "Remote Mac" : "Unavailable",
            online: snapshot?.isOnline == true,
            selected: selected,
            active: snapshot?.activeRequests ?? 0,
            queued: snapshot?.queuedRequests ?? 0,
            models: snapshot?.catalog.filter(\.cached).count ?? 0,
            loaded: snapshot?.loadedModels.count ?? 0,
            latency: snapshot?.roundTripSeconds ?? 0,
            useAction: useAction,
            removeAction: { appState.removeTeamHost(host) }
        )
    }

    private func hostNode(
        name: String,
        subtitle: String,
        online: Bool,
        selected: Bool,
        active: Int,
        queued: Int,
        models: Int,
        loaded: Int,
        latency: Double,
        useAction: (() -> Void)?,
        removeAction: (() -> Void)?
    ) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 9) {
                Image(systemName: "desktopcomputer")
                    .foregroundStyle(selected ? Color.green : Color.secondary)
                    .frame(width: 28, height: 28)
                    .background(selected ? Color.green.opacity(0.12) : Color.secondary.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                VStack(alignment: .leading, spacing: 2) {
                    Text(name)
                        .font(.body.weight(.semibold))
                        .lineLimit(1)
                    HStack(spacing: 5) {
                        statusIndicator(active: online)
                        Text(subtitle)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
                Spacer(minLength: 4)
                if selected {
                    Text("LAST ROUTE")
                        .font(.caption2.weight(.bold))
                        .foregroundStyle(.green)
                }
            }

            HStack(spacing: 16) {
                compactMetric("Active", value: active)
                compactMetric("Queued", value: queued)
                compactMetric("Ready", value: models)
                compactMetric("Loaded", value: loaded)
                if latency > 0 {
                    compactLatency(latency)
                }
            }

            HStack {
                if let useAction {
                    Button(
                        appState.inferenceMode == .local ? "Enable Pool" : "Use This Mac",
                        action: useAction
                    )
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                } else {
                    Text(selected ? "Handled the last request" : "Available to auto-route")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if let removeAction {
                    Button(role: .destructive, action: removeAction) {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.borderless)
                    .help("Forget device")
                }
            }
        }
        .padding(14)
        .frame(width: 300, height: 146, alignment: .topLeading)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(selected ? Color.green.opacity(0.7) : Color(nsColor: .separatorColor), lineWidth: 1)
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
                Text("No other MachBoost devices found yet. LAN access must be enabled on the host. This Mac is intentionally hidden from this list.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            } else {
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 250, maximum: 360), spacing: 12)],
                    alignment: .leading,
                    spacing: 12
                ) {
                    ForEach(nearbyHosts) { host in
                        VStack(alignment: .leading, spacing: 12) {
                            HStack(spacing: 10) {
                                Image(systemName: "desktopcomputer.and.arrow.down")
                                    .foregroundStyle(.green)
                                    .frame(width: 28, height: 28)
                                    .background(Color.green.opacity(0.1))
                                    .clipShape(RoundedRectangle(cornerRadius: 6))
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(host.name)
                                        .font(.body.weight(.semibold))
                                        .lineLimit(1)
                                    Text(host.version.map { "MachBoost \($0)" } ?? "MachBoost host")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            Text(host.endpoint.absoluteString)
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .textSelection(.enabled)
                            Button {
                                pendingNearbyHost = host
                            } label: {
                                Label("Connect", systemImage: "link.badge.plus")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.green)
                        }
                        .padding(14)
                        .background(Color(nsColor: .controlBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .overlay {
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
                        }
                    }
                }
            }
        }
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
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 10) {
                        modelRequestFields
                    }
                    VStack(alignment: .leading, spacing: 10) {
                        modelRequestFields
                    }
                }
            }
        }
    }

    private var nearbyHosts: [DiscoveredMachBoostHost] {
        let saved = Set(appState.teamHosts.map { $0.endpoint.absoluteString })
        return appState.hostDiscovery.hosts.filter {
            $0.deviceID != appState.deviceID && !saved.contains($0.endpoint.absoluteString)
        }
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

    private func compactMetric(_ label: String, value: Int) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text("\(value)")
                .font(.callout.weight(.medium))
                .monospacedDigit()
                .lineLimit(1)
        }
    }

    private func compactLatency(_ seconds: Double) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("RTT")
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text("\(Int((seconds * 1_000).rounded())) ms")
                .font(.callout.weight(.medium))
                .monospacedDigit()
                .lineLimit(1)
        }
    }

    @ViewBuilder
    private var modelRequestFields: some View {
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
