import Foundation
import SwiftUI

struct AppsView: View {
    @Environment(AppState.self) private var appState
    @State private var selectedSource = "local"
    @State private var status = ClaudeDesktopConnectionStatus.current()
    @State private var pendingAction: ClaudeDesktopAction?
    @State private var isWorking = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                header
                claudeDesktopSection
            }
            .frame(maxWidth: 820, alignment: .leading)
            .padding(28)
        }
        .navigationTitle("Apps")
        .confirmationDialog(
            pendingAction?.title ?? "",
            isPresented: Binding(
                get: { pendingAction != nil },
                set: { if !$0 { pendingAction = nil } }
            ),
            titleVisibility: .visible
        ) {
            if let pendingAction {
                Button(pendingAction.buttonTitle, role: pendingAction.role) {
                    Task { await apply(pendingAction) }
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Claude Desktop will restart. Any running Claude task will stop.")
        }
        .onAppear(perform: selectCurrentSource)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Use MachBoost Anywhere")
                .font(.title2.weight(.semibold))
            Text("Connect desktop agents to models running on this Mac or a shared MachBoost host.")
                .foregroundStyle(.secondary)
        }
    }

    private var claudeDesktopSection: some View {
        VStack(spacing: 0) {
            HStack(spacing: 14) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.green.opacity(0.13))
                    Image(systemName: "sparkles.rectangle.stack")
                        .font(.title2)
                        .foregroundStyle(.green)
                }
                .frame(width: 46, height: 46)

                VStack(alignment: .leading, spacing: 3) {
                    Text("Claude Desktop")
                        .font(.headline)
                    Text(status.connected ? "Connected through MachBoost" : "Third-party inference gateway")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                if isWorking {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Toggle(
                        "Use MachBoost models in Claude Desktop",
                        isOn: Binding(
                            get: { status.connected },
                            set: { enabled in
                                pendingAction = enabled ? .connect : .disconnect
                            }
                        )
                    )
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .accessibilityIdentifier("claude-desktop-toggle")
                }
            }
            .padding(18)

            Divider()

            VStack(alignment: .leading, spacing: 14) {
                LabeledContent("Inference host") {
                    Picker("Inference host", selection: $selectedSource) {
                        Label("This Mac", systemImage: "desktopcomputer")
                            .tag("local")
                        ForEach(appState.teamHosts) { host in
                            Label(host.hostName, systemImage: "network")
                                .tag(host.id.uuidString)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .frame(maxWidth: 320, alignment: .trailing)
                    .accessibilityIdentifier("claude-desktop-host-picker")
                }

                LabeledContent("Gateway") {
                    Text(selectedEndpoint.absoluteString)
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                }

                if let currentEndpoint = status.endpoint, status.connected {
                    LabeledContent("Active connection") {
                        HStack(spacing: 6) {
                            if status.relayed {
                                Image(systemName: "lock.shield.fill")
                                    .foregroundStyle(.green)
                                    .help("Claude connects through a private localhost bridge")
                            }
                            Text(status.upstream ?? currentEndpoint)
                                .foregroundStyle(.green)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                    }
                }

                HStack(spacing: 8) {
                    Image(systemName: "shippingbox.fill")
                        .foregroundStyle(.secondary)
                    Text("Claude discovers up to five available models from the selected host, including shared MLX models.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
            }
            .padding(18)
        }
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
        }
    }

    private var selectedEndpoint: URL {
        guard
            selectedSource != "local",
            let id = UUID(uuidString: selectedSource),
            let host = appState.teamHosts.first(where: { $0.id == id })
        else {
            return appState.configuration.endpoint
        }
        return host.endpoint
    }

    private func selectCurrentSource() {
        status = ClaudeDesktopConnectionStatus.current()
        guard let endpoint = status.upstream ?? status.endpoint else { return }
        if endpoint == appState.configuration.endpoint.absoluteString {
            selectedSource = "local"
        } else if let host = appState.teamHosts.first(where: {
            $0.endpoint.absoluteString == endpoint
        }) {
            selectedSource = host.id.uuidString
        }
    }

    private func apply(_ action: ClaudeDesktopAction) async {
        pendingAction = nil
        isWorking = true
        defer { isWorking = false }
        do {
            switch action {
            case .connect:
                let token = try await selectedToken()
                _ = try await appState.daemon.runCLI(
                    [
                        "launch",
                        "claude-desktop",
                        "--endpoint",
                        selectedEndpoint.absoluteString,
                        "--yes",
                    ],
                    apiToken: token
                )
            case .disconnect:
                _ = try await appState.daemon.runCLI(
                    ["launch", "claude-desktop", "--restore", "--yes"]
                )
            }
            status = ClaudeDesktopConnectionStatus.current()
        } catch {
            appState.presentedError = error.localizedDescription
            status = ClaudeDesktopConnectionStatus.current()
        }
    }

    private func selectedToken() async throws -> String? {
        if selectedSource == "local" {
            let keychainToken = appState.daemon.authenticationRequired
                ? await Task.detached(priority: .userInitiated) {
                    KeychainStore.token()
                }.value
                : nil
            return try AppsGatewayCredentials.localToken(
                authenticationRequired: appState.daemon.authenticationRequired,
                runtimeToken: appState.apiToken,
                keychainToken: keychainToken
            )
        }
        guard
            let id = UUID(uuidString: selectedSource),
            let token = await KeychainStore.teamTokenAsync(profileID: id),
            !token.isEmpty
        else {
            throw AppsViewError.missingTeamToken
        }
        return token
    }
}

enum AppsGatewayCredentials {
    static func localToken(
        authenticationRequired: Bool,
        runtimeToken: String?,
        keychainToken: String?
    ) throws -> String? {
        guard authenticationRequired else { return nil }
        if let runtimeToken, !runtimeToken.isEmpty {
            return runtimeToken
        }
        if let keychainToken, !keychainToken.isEmpty {
            return keychainToken
        }
        throw AppsViewError.missingLocalToken
    }
}

private enum ClaudeDesktopAction {
    case connect
    case disconnect

    var title: String {
        switch self {
        case .connect: "Connect Claude Desktop to MachBoost?"
        case .disconnect: "Disconnect Claude Desktop from MachBoost?"
        }
    }

    var buttonTitle: String {
        switch self {
        case .connect: "Connect and Restart"
        case .disconnect: "Disconnect and Restart"
        }
    }

    var role: ButtonRole? {
        switch self {
        case .connect: nil
        case .disconnect: .destructive
        }
    }
}

private struct ClaudeDesktopConnectionStatus {
    let connected: Bool
    let endpoint: String?
    let upstream: String?
    let relayed: Bool

    static func current() -> Self {
        let support = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first
        let library = support?
            .appendingPathComponent("Claude-3p", isDirectory: true)
            .appendingPathComponent("configLibrary", isDirectory: true)
        guard let library else {
            return Self(connected: false, endpoint: nil, upstream: nil, relayed: false)
        }
        let meta = json(at: library.appendingPathComponent("_meta.json"))
        let profile = json(
            at: library.appendingPathComponent(
                "00000000-0000-4000-8000-000000000135.json"
            )
        )
        let connected = meta["appliedId"] as? String
            == "00000000-0000-4000-8000-000000000135"
            && profile["inferenceProvider"] as? String == "gateway"
        let endpoint = connected ? profile["inferenceGatewayBaseUrl"] as? String : nil
        let relay = json(
            at: FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".machboost", isDirectory: true)
                .appendingPathComponent("claude-loopback-relay.json")
        )
        let relayed = connected
            && relay["schema"] as? String == "machboost.claude-loopback-relay.v1"
            && relay["endpoint"] as? String == endpoint
        return Self(
            connected: connected,
            endpoint: endpoint,
            upstream: relayed ? relay["upstream"] as? String : endpoint,
            relayed: relayed
        )
    }

    private static func json(at url: URL) -> [String: Any] {
        guard
            let data = try? Data(contentsOf: url),
            let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return [:]
        }
        return value
    }
}

private enum AppsViewError: LocalizedError {
    case missingLocalToken
    case missingTeamToken

    var errorDescription: String? {
        switch self {
        case .missingLocalToken:
            "This Mac requires authentication, but its MachBoost API key is missing. Turn LAN sharing off and on in Server settings to create a new key."
        case .missingTeamToken:
            "The saved API key for this MachBoost host is missing. Reconnect the host first."
        }
    }
}
