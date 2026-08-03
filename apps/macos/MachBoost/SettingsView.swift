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
                    Toggle("Automatically check for updates", isOn: $automaticUpdates)
                        .accessibilityIdentifier("automatic-updates")
                        .onChange(of: automaticUpdates) {
                            updates.automaticallyChecksForUpdates = automaticUpdates
                        }
                        .disabled(!updates.supportsAutomaticUpdates)
                    LabeledContent("Updates") {
                        Text(updates.deliveryDescription)
                            .foregroundStyle(.secondary)
                    }
                    LabeledContent("Version") {
                        Text(version)
                            .foregroundStyle(.secondary)
                    }
                    Button {
                        updates.checkForUpdates()
                    } label: {
                        Label(updates.actionTitle, systemImage: "arrow.down.circle")
                    }
                    .disabled(!updates.isAvailable)
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
