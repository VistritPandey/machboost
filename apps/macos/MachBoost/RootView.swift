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
                .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 340)
        } detail: {
            detail
        }
        .frame(minWidth: 980, minHeight: 680)
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
                Text(appState.serverIsRunning ? "Server ready" : "Server offline")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(appState.loadedModels.count) loaded")
                    .font(.caption)
                    .foregroundStyle(.secondary)
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
        let defaultModel = appState.catalog.first(where: { $0.cached && $0.recommended })?.name
            ?? appState.catalog.first(where: \.cached)?.name
            ?? "llama3.2:3b"
        let conversation = Conversation(model: defaultModel)
        modelContext.insert(conversation)
        try? modelContext.save()
        selection = .conversation(conversation.id)
    }

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
        panel.allowedContentTypes = [.markdown]
        panel.nameFieldStringValue = "\(sanitized(conversation.title)).md"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        var lines = ["# \(conversation.title)", "", "Model: `\(conversation.model)`", ""]
        if !conversation.attachments.isEmpty {
            lines.append("Context: \(conversation.orderedAttachments.map(\.displayName).joined(separator: ", "))")
            lines.append("")
        }
        for message in conversation.orderedMessages {
            lines.append("## \(message.role == .user ? "User" : "Assistant")")
            lines.append("")
            lines.append(message.content)
            lines.append("")
        }
        do {
            try lines.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
        } catch {
            appState.presentedError = error.localizedDescription
        }
    }

    private func sanitized(_ name: String) -> String {
        name.replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: ":", with: "-")
    }
}
