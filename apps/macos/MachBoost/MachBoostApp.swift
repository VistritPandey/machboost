import AppKit
import SwiftData
import SwiftUI

@main
struct MachBoostApp: App {
    @NSApplicationDelegateAdaptor(MachBoostAppDelegate.self)
    private var appDelegate
    @State private var appState: AppState
    @StateObject private var updates: UpdateController
    private let modelContainer: ModelContainer

    init() {
        let appState = AppState()
        _appState = State(initialValue: appState)
        _updates = StateObject(wrappedValue: UpdateController())
        do {
            modelContainer = try ModelContainer(
                for: Conversation.self,
                ChatMessage.self,
                ChatAttachment.self
            )
        } catch {
            fatalError("Could not open the local MachBoost chat store: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup(id: "main") {
            RootView(updates: updates)
                .environment(appState)
                .task {
                    appDelegate.appState = appState
                    await appState.start()
                }
        }
        .modelContainer(modelContainer)
        .defaultSize(width: 1_180, height: 780)
        .commands {
            CommandGroup(after: .appInfo) {
                Button("Check for Updates…") {
                    updates.checkForUpdates()
                }
            }
        }

        MenuBarExtra {
            MenuBarContent()
                .environment(appState)
        } label: {
            Label(
                "MachBoost",
                systemImage: appState.serverIsRunning ? "bolt.fill" : "bolt.slash"
            )
        }
        .menuBarExtraStyle(.menu)

        Settings {
            SettingsView(updates: updates)
                .environment(appState)
                .frame(width: 680, height: 540)
        }
    }
}

private struct MenuBarContent: View {
    @Environment(AppState.self) private var appState
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button("Open MachBoost") {
            openWindow(id: "main")
            NSApplication.shared.activate(ignoringOtherApps: true)
        }

        Divider()

        Label(
            appState.serverIsRunning ? "Server running" : "Server stopped",
            systemImage: appState.serverIsRunning ? "checkmark.circle.fill" : "xmark.circle"
        )

        if appState.loadedModels.isEmpty {
            Text("No loaded models")
        } else {
            Menu("Loaded models") {
                ForEach(appState.loadedModels) { model in
                    Button {
                        Task { await appState.stop(model: model.model) }
                    } label: {
                        Label(model.model, systemImage: "eject")
                    }
                }
            }
        }

        Button("Unload all models") {
            Task { await appState.pauseServing() }
        }
        .disabled(appState.loadedModels.isEmpty)

        Divider()

        Button("Quit MachBoost") {
            NSApplication.shared.terminate(nil)
        }
        .keyboardShortcut("q")
    }
}

@MainActor
final class MachBoostAppDelegate: NSObject, NSApplicationDelegate {
    weak var appState: AppState?
    private var terminating = false

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard
            !terminating,
            let appState,
            appState.daemon.state != .stopped
        else {
            return .terminateNow
        }
        terminating = true
        Task {
            await appState.shutdown()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }

    func applicationWillTerminate(_ notification: Notification) {
        appState?.daemon.terminateImmediately()
    }
}
