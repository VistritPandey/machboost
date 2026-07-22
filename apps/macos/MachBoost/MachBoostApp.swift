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
            let schema = Schema([
                Conversation.self,
                ChatMessage.self,
                ChatAttachment.self,
            ])
            let isUITesting = ProcessInfo.processInfo.environment["MACHBOOST_UI_TESTING"] == "1"
            let configuration = ModelConfiguration(
                schema: schema,
                isStoredInMemoryOnly: isUITesting
            )
            modelContainer = try ModelContainer(for: schema, configurations: configuration)
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
                    let environment = ProcessInfo.processInfo.environment
                    if environment["MACHBOOST_UI_TESTING"] == "1" {
                        await appState.startUITestMode()
                    } else if
                        environment["MACHBOOST_TESTING"] != "1",
                        environment["XCTestConfigurationFilePath"] == nil
                    {
                        await appState.start()
                    }
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

        if appState.serverIsRunning {
            Button("Pause serving") {
                Task { await appState.pauseServer() }
            }
        } else {
            Button("Resume serving") {
                Task { await appState.resumeServer() }
            }
        }

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
            Task { await appState.unloadAllModels() }
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
