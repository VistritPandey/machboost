import AppKit
import ServiceManagement
import SwiftUI

struct SettingsView: View {
    @Environment(AppState.self) private var appState
    @ObservedObject var updates: UpdateController
    @State private var launchAtLogin = false
    @State private var automaticUpdates = true

    var body: some View {
        ScrollView {
            Form {
                Section("General") {
                    Toggle("Launch MachBoost at login", isOn: $launchAtLogin)
                        .accessibilityIdentifier("launch-at-login")
                        .onChange(of: launchAtLogin, updateLoginItem)
                }

                Section("Updates") {
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: updateIcon)
                            .font(.title2)
                            .foregroundStyle(updateColor)
                            .frame(width: 28)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(updates.deliveryDescription)
                                .font(.body.weight(.medium))
                            Text(updateDetail)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text("v\(version)")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }

                    Toggle("Check for new releases automatically", isOn: $automaticUpdates)
                        .accessibilityIdentifier("automatic-updates")
                        .onChange(of: automaticUpdates) {
                            updates.automaticallyChecksForUpdates = automaticUpdates
                        }
                        .disabled(!updates.supportsAutomaticUpdates)

                    HStack {
                        Button {
                            updates.checkForUpdates()
                        } label: {
                            if updates.isChecking {
                                ProgressView()
                                    .controlSize(.small)
                            } else {
                                Label(updates.actionTitle, systemImage: "arrow.clockwise")
                            }
                        }
                        .disabled(!updates.isAvailable || updates.isChecking)
                        .accessibilityIdentifier("check-for-updates")

                        if updates.canDownloadUpdate {
                            Button {
                                updates.downloadUpdate()
                            } label: {
                                Label(updates.downloadTitle, systemImage: "arrow.down.circle")
                            }
                            .buttonStyle(.borderedProminent)
                            .accessibilityIdentifier("download-update")
                        }
                    }
                }

                Section("Storage") {
                    storageRow(
                        title: "Models and cache",
                        url: FileManager.default.homeDirectoryForCurrentUser
                            .appendingPathComponent(".cache", isDirectory: true)
                    )
                    storageRow(
                        title: "Chats and attachments",
                        url: applicationSupportURL
                    )
                }

                Section("Privacy") {
                    LabeledContent("Chat history") {
                        Text("Stored on this Mac")
                            .foregroundStyle(.secondary)
                    }
                    LabeledContent("Telemetry") {
                        Text("Disabled")
                            .foregroundStyle(.secondary)
                    }
                    LabeledContent("LAN API") {
                        Text(appState.configuration.lanEnabled ? "Authenticated" : "Off")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .formStyle(.grouped)
            .padding(20)
            .frame(maxWidth: 720)
            .frame(maxWidth: .infinity)
        }
        .navigationTitle("Settings")
        .onAppear {
            launchAtLogin = SMAppService.mainApp.status == .enabled
            automaticUpdates = updates.automaticallyChecksForUpdates
        }
    }

    private var version: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
            ?? "Development"
    }

    private var updateIcon: String {
        if updates.isChecking { return "arrow.triangle.2.circlepath" }
        if updates.updateAvailable { return "arrow.down.circle.fill" }
        if updates.communityCheckFailed { return "exclamationmark.triangle.fill" }
        return "checkmark.circle.fill"
    }

    private var updateColor: Color {
        if updates.updateAvailable { return .green }
        if updates.communityCheckFailed { return .orange }
        return .secondary
    }

    private var updateDetail: String {
        if updates.supportsAutomaticUpdates {
            if updates.canDownloadUpdate {
                return "Community builds check automatically; download and approval remain manual."
            }
            if let date = updates.lastCheckedAt {
                return "Last checked \(date.formatted(date: .abbreviated, time: .shortened))."
            }
            return "Automatic checks are enabled by default. Signed builds install with Sparkle."
        }
        return "Update checks are unavailable in this build."
    }

    private var applicationSupportURL: URL {
        let root = try? FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return (root ?? FileManager.default.homeDirectoryForCurrentUser)
            .appendingPathComponent("MachBoost", isDirectory: true)
    }

    private func storageRow(title: String, url: URL) -> some View {
        LabeledContent(title) {
            HStack {
                Text(url.path)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Button {
                    NSWorkspace.shared.activateFileViewerSelecting([url])
                } label: {
                    Image(systemName: "folder")
                }
                .help("Show in Finder")
            }
        }
    }

    private func updateLoginItem() {
        do {
            if launchAtLogin {
                try SMAppService.mainApp.register()
            } else {
                try SMAppService.mainApp.unregister()
            }
        } catch {
            launchAtLogin = SMAppService.mainApp.status == .enabled
            appState.presentedError = error.localizedDescription
        }
    }
}
