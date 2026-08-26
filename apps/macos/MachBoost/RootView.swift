import AppKit
import SwiftData
import SwiftUI
import UniformTypeIdentifiers

struct RootView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Conversation.updatedAt, order: .reverse)
    private var conversations: [Conversation]

    @ObservedObject var updates: UpdateController
    @State private var selection: SidebarDestination?
    @State private var search = ""
    @State private var renaming: Conversation?
    @State private var renameValue = ""

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 190, ideal: 230, max: 300)
        } detail: {
            detail
        }
        .frame(minWidth: 760, minHeight: 560)
        .onAppear {
            DispatchQueue.main.async {
                constrainWindowToVisibleScreen()
            }
        }
        .task {
            if conversations.isEmpty {
                newConversation()
            } else if selection == nil, let first = conversations.first {
                selection = .conversation(first.id)
            }
        }
        .sheet(isPresented: onboardingBinding) {
            ModelOnboardingView()
        }
        .alert("Rename conversation", isPresented: renameBinding) {
            TextField("Name", text: $renameValue)
            Button("Rename") { finishRename() }
            Button("Cancel", role: .cancel) { renaming = nil }
        }
        .alert("MachBoost", isPresented: errorBinding) {
            Button("OK") { appState.presentedError = nil }
        } message: {
            Text(appState.presentedError ?? "Unknown error")
        }
    }

    private func constrainWindowToVisibleScreen() {
        guard
            let window = NSApplication.shared.keyWindow
                ?? NSApplication.shared.windows.first(where: { $0.isVisible }),
            !window.styleMask.contains(.fullScreen),
            let screen = window.screen ?? NSScreen.main
        else {
            return
        }
        let visible = screen.visibleFrame
        var frame = window.frame
        frame.size.width = min(frame.width, visible.width)
        frame.size.height = min(frame.height, visible.height)
        frame.origin.x = min(max(frame.minX, visible.minX), visible.maxX - frame.width)
        frame.origin.y = min(max(frame.minY, visible.minY), visible.maxY - frame.height)
        guard frame != window.frame else { return }
        window.setFrame(frame, display: true)
    }

    private var sidebar: some View {
        VStack(spacing: 0) {
            HStack {
                Text("MachBoost")
                    .font(.headline)
                Spacer()
                Button(action: newConversation) {
                    Image(systemName: "square.and.pencil")
                }
                .buttonStyle(.plain)
                .accessibilityLabel("New chat")
                .help("New chat")
            }
            .padding(.horizontal, 12)
            .frame(height: 44)

            Divider()

            List(selection: $selection) {
                Section("Chats") {
                    ForEach(filteredConversations) { conversation in
                        Label(conversation.title, systemImage: "bubble.left")
                            .lineLimit(1)
                            .tag(SidebarDestination.conversation(conversation.id))
                            .contextMenu {
                                Button("Rename") { beginRename(conversation) }
                                Button("Export") { export(conversation) }
                                Divider()
                                Button("Delete", role: .destructive) {
                                    delete(conversation)
                                }
                            }
                    }
                }

                Section("Workspace") {
                    Label("Apps", systemImage: "square.grid.2x2")
                        .tag(SidebarDestination.apps)
                    Label("Connections", systemImage: "point.3.connected.trianglepath.dotted")
                        .tag(SidebarDestination.connections)
                    Label("Models", systemImage: "shippingbox")
                        .tag(SidebarDestination.models)
                    Label("Server", systemImage: "server.rack")
                        .tag(SidebarDestination.server)
                    Label("Settings", systemImage: "gearshape")
                        .tag(SidebarDestination.settings)
                }
            }
            .listStyle(.sidebar)
            .searchable(text: $search, placement: .sidebar, prompt: "Search chats")

            Divider()

            HStack(spacing: 7) {
                Circle()
                    .fill(appState.serverIsRunning ? Color.green : Color.red)
                    .frame(width: 8, height: 8)
                Text(appState.inferenceMode == .team ? appState.inferenceLabel : (appState.serverIsRunning ? "Local ready" : "Local offline"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer()
                Text("\(appState.activeLoadedModels.count) loaded")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .padding(12)
        }
    }

    @ViewBuilder
    private var detail: some View {
        switch selection {
        case let .conversation(id):
            if let conversation = conversations.first(where: { $0.id == id }) {
                ChatView(conversation: conversation)
            } else {
                emptyDetail
            }
        case .apps:
            AppsView()
        case .connections:
            ConnectionsView()
        case .models:
            ModelsView()
        case .server:
            ServerView()
        case .settings:
            SettingsView(updates: updates)
        case nil:
            emptyDetail
        }
    }

    private var emptyDetail: some View {
        ContentUnavailableView(
            "Select a chat",
            systemImage: "bubble.left.and.bubble.right"
        )
    }

    private var filteredConversations: [Conversation] {
        let query = search.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return conversations }
        return conversations.filter { conversation in
            conversation.title.localizedCaseInsensitiveContains(query)
                || conversation.messages.contains {
                    $0.content.localizedCaseInsensitiveContains(query)
                }
        }
    }

    private var onboardingBinding: Binding<Bool> {
        Binding(
            get: { appState.showOnboarding },
            set: { appState.showOnboarding = $0 }
        )
    }

    private var errorBinding: Binding<Bool> {
        Binding(
            get: { appState.presentedError != nil },
            set: { if !$0 { appState.presentedError = nil } }
        )
    }

    private var renameBinding: Binding<Bool> {
        Binding(
            get: { renaming != nil },
            set: { if !$0 { renaming = nil } }
        )
    }

    private func newConversation() {
        let defaultModel = uiTestModel
            ?? appState.activeCatalog.first(where: { $0.cached && $0.recommended })?.name
            ?? appState.activeCatalog.first(where: \.cached)?.name
            ?? "llama3.2:3b"
        let conversation = Conversation(model: defaultModel)
        modelContext.insert(conversation)
#if DEBUG
        seedUITestAttachments(into: conversation)
#endif
        try? modelContext.save()
        selection = .conversation(conversation.id)
    }

    private var uiTestModel: String? {
#if DEBUG
        guard ProcessInfo.processInfo.environment["MACHBOOST_UI_TESTING"] == "1" else {
            return nil
        }
        return ProcessInfo.processInfo.environment["MACHBOOST_UI_TEST_MODEL"]
#else
        return nil
#endif
    }

#if DEBUG
    private func seedUITestAttachments(into conversation: Conversation) {
        guard ProcessInfo.processInfo.environment["MACHBOOST_UI_TESTING"] == "1" else {
            return
        }
        let requested = Set(
            ProcessInfo.processInfo.environment["MACHBOOST_UI_TEST_ATTACHMENTS", default: ""]
                .split(separator: ",")
                .map(String.init)
        )
        let fixtures: [(String, AttachmentKind, String)] = [
            ("text", .text, "fixture-context.txt"),
            ("image", .image, "fixture-image.png"),
        ]
        for (key, kind, name) in fixtures where requested.contains(key) {
            let attachment = ChatAttachment(
                kind: kind,
                displayName: name,
                importedPath: "/tmp/machboost-ui-fixture/\(name)",
                sourcePath: "/tmp/machboost-ui-fixture/\(name)",
                byteCount: 128,
                conversation: conversation
            )
            modelContext.insert(attachment)
            conversation.attachments.append(attachment)
        }
    }
#endif

    private func beginRename(_ conversation: Conversation) {
        renaming = conversation
        renameValue = conversation.title
    }

    private func finishRename() {
        guard let renaming else { return }
        let value = renameValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if !value.isEmpty {
            renaming.title = value
            renaming.updatedAt = .now
            try? modelContext.save()
        }
        self.renaming = nil
    }

    private func delete(_ conversation: Conversation) {
        for attachment in conversation.attachments {
            AttachmentStore.remove(attachment)
        }
        modelContext.delete(conversation)
        try? modelContext.save()
        if selection == .conversation(conversation.id) {
            selection = conversations.first(where: { $0.id != conversation.id }).map {
                .conversation($0.id)
            }
        }
    }

    private func export(_ conversation: Conversation) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [UTType(filenameExtension: "md") ?? .plainText]
        panel.nameFieldStringValue = ConversationExporter.fileName(for: conversation)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try ConversationExporter.markdown(conversation).write(
                to: url,
                atomically: true,
                encoding: .utf8
            )
        } catch {
            appState.presentedError = error.localizedDescription
        }
    }
}
