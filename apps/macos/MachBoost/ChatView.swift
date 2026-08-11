import AppKit
import SwiftData
import SwiftUI
import UniformTypeIdentifiers

struct ChatView: View {
    private static let bottomAnchor = "machboost-chat-bottom"

    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @Bindable var conversation: Conversation

    @State private var draft = ""
    @State private var generationTask: Task<Void, Never>?
    @State private var activeRequestID: String?
    @State private var activeAssistant: ChatMessage?
    @State private var isImporting = false
    @State private var isAddingWorkspace = false
    @State private var showsGenerationControls = false
    @FocusState private var composerIsFocused: Bool
    @AppStorage("machboost.chat.maxTokens") private var maxTokens = 512
    @AppStorage("machboost.chat.temperature") private var temperature = 0.2
    @AppStorage("machboost.chat.reasoningStrength") private var reasoningStrength = "medium"
    @AppStorage("machboost.chat.showReasoning") private var showReasoning = true

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            messageList
            Divider()
            contextStrip
            composer
        }
        .fileImporter(
            isPresented: $isImporting,
            allowedContentTypes: [.image, .plainText, .sourceCode, .folder],
            allowsMultipleSelection: true,
            onCompletion: importAttachments
        )
        .onAppear {
            selectAvailableModelIfNeeded()
        }
        .onChange(of: selectableModels.map(\.name)) {
            selectAvailableModelIfNeeded()
        }
        .onDisappear {
            stop()
        }
    }

    private var header: some View {
        HStack(spacing: 12) {
            Picker("Model", selection: $conversation.model) {
                ForEach(selectableModels) { model in
                    Label(
                        model.displayName,
                        systemImage: model.supportsVision ? "eye" : "text.bubble"
                    )
                    .tag(model.name)
                }
                if !selectableModels.contains(where: { $0.name == conversation.model }) {
                    Text(conversation.model).tag(conversation.model)
                }
            }
            .labelsHidden()
            .frame(maxWidth: 280)

            if let model = appState.model(named: conversation.model) {
                Text(model.backend.uppercased())
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
                capabilityIcons(for: model)
            }

            workspaceMenu

            Spacer()

            Button {
                showsGenerationControls.toggle()
            } label: {
                Image(systemName: "slider.horizontal.3")
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Generation controls")
            .help("Generation controls")
            .popover(isPresented: $showsGenerationControls, arrowEdge: .bottom) {
                generationControls
            }

            if let activeRequestID {
                Text(activeRequestID.suffix(8))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 16)
        .frame(height: 48)
    }

    private var workspaceMenu: some View {
        Menu {
            Button {
                conversation.workspaceID = nil
                try? modelContext.save()
            } label: {
                Label("No Repository", systemImage: "minus.circle")
            }

            if !appState.workspaces.isEmpty {
                Divider()
                ForEach(appState.workspaces) { workspace in
                    Button {
                        conversation.workspaceID = workspace.id
                        try? modelContext.save()
                    } label: {
                        if conversation.workspaceID == workspace.id {
                            Label(workspace.name, systemImage: "checkmark")
                        } else {
                            Text(workspace.name)
                        }
                    }
                }
            }

            Divider()
            Button(action: chooseWorkspace) {
                Label("Open Repository...", systemImage: "folder.badge.plus")
            }

            if let workspace = selectedWorkspace {
                Button {
                    Task {
                        await appState.reindexWorkspace(id: workspace.id)
                    }
                } label: {
                    Label("Refresh Index", systemImage: "arrow.clockwise")
                }
                .disabled(appState.indexingWorkspaces.contains(workspace.id))

                Button {
                    conversation.workspaceID = nil
                    try? modelContext.save()
                    Task {
                        await appState.removeWorkspace(id: workspace.id)
                    }
                } label: {
                    Label("Remove Repository", systemImage: "trash")
                }
            }
        } label: {
            HStack(spacing: 6) {
                if isAddingWorkspace
                    || selectedWorkspace.map({
                        appState.indexingWorkspaces.contains($0.id)
                    }) == true {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    Image(systemName: "folder")
                }
                Text(selectedWorkspace?.name ?? "Repository")
                    .lineLimit(1)
                if let workspace = selectedWorkspace {
                    Text("\(workspace.fileCount)")
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
            }
        }
        .menuStyle(.borderlessButton)
        .accessibilityIdentifier("repository-picker")
        .fixedSize(horizontal: true, vertical: false)
        .help(workspaceHelp)
    }

    private var generationControls: some View {
        Form {
            Stepper(value: $maxTokens, in: 32...4_096, step: 32) {
                LabeledContent("Maximum tokens", value: "\(maxTokens)")
            }
            VStack(alignment: .leading, spacing: 6) {
                LabeledContent(
                    "Temperature",
                    value: temperature.formatted(.number.precision(.fractionLength(2)))
                )
                Slider(value: $temperature, in: 0...1, step: 0.05)
            }
            if selectedModel?.supportsReasoning == true {
                Picker("Reasoning", selection: $reasoningStrength) {
                    Text("Off").tag("off")
                    Text("Low").tag("low")
                    Text("Medium").tag("medium")
                    Text("High").tag("high")
                    Text("Max").tag("xhigh")
                }
                .pickerStyle(.segmented)
                Toggle("Show reasoning", isOn: $showReasoning)
            }
            if let contextLength = selectedModel?.contextLength {
                LabeledContent("Context window", value: contextLength.formatted())
            }
        }
        .formStyle(.grouped)
        .frame(width: 300)
        .padding(.vertical, 6)
    }

    private var messageList: some View {
        let messages: [ChatMessage] = conversation.orderedMessages

        return ScrollViewReader { (proxy: ScrollViewProxy) in
            ScrollView(.vertical, showsIndicators: true) {
                LazyVStack(alignment: .leading, spacing: 0) {
                    if messages.isEmpty {
                        ContentUnavailableView(
                            "Start a conversation",
                            systemImage: "bubble.left.and.text.bubble.right",
                            description: Text(conversation.model)
                        )
                        .frame(maxWidth: .infinity, minHeight: 360)
                    } else {
                        ForEach(messages, id: \.id) { message in
                            MessageRow(
                                message: message,
                                showsReasoning: showReasoning,
                                onEdit: { edit(message) },
                                onRegenerate: regenerateAction(
                                    for: message,
                                    in: messages
                                )
                            )
                            .id(message.id)
                        }
                    }

                    Color.clear
                        .frame(height: 1)
                        .id(Self.bottomAnchor)
                        .accessibilityIdentifier("chat-scroll-bottom")
                }
                .padding(.vertical, 10)
            }
            .defaultScrollAnchor(.bottom)
            .onAppear {
                proxy.scrollTo(Self.bottomAnchor, anchor: .bottom)
            }
            .onChange(of: conversation.id) {
                proxy.scrollTo(Self.bottomAnchor, anchor: .bottom)
            }
            .onChange(of: messages.count) {
                withAnimation(.easeOut(duration: 0.18)) {
                    proxy.scrollTo(Self.bottomAnchor, anchor: .bottom)
                }
            }
            .task(
                id: [
                    messages.last?.content ?? "",
                    messages.last?.reasoningContent ?? "",
                    messages.last?.toolCallsJSON ?? "",
                ].joined(separator: "|")
            ) {
                try? await Task.sleep(for: .milliseconds(100))
                guard !Task.isCancelled else { return }
                proxy.scrollTo(Self.bottomAnchor, anchor: .bottom)
            }
        }
    }

    @ViewBuilder
    private var contextStrip: some View {
        if !conversation.orderedAttachments.isEmpty {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    ForEach(conversation.orderedAttachments) { attachment in
                        HStack(spacing: 6) {
                            Image(systemName: attachment.kind == .image ? "photo" : "doc.text")
                            Text(attachment.displayName)
                                .lineLimit(1)
                            Button {
                                remove(attachment)
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                            }
                            .buttonStyle(.plain)
                            .help("Remove context")
                        }
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .frame(height: 28)
                        .background(Color(nsColor: .controlBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                    }
                }
                .padding(.horizontal, 12)
                .padding(.top, 8)
            }
        }
    }

    private var composer: some View {
        HStack(alignment: .bottom, spacing: 8) {
            Button {
                isImporting = true
            } label: {
                Image(systemName: "paperclip")
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.borderless)
            .accessibilityLabel("Attach files")
            .help("Attach text, code, folder, or image")

            TextField("Message MachBoost", text: $draft, axis: .vertical)
                .focused($composerIsFocused)
                .textFieldStyle(.plain)
                .lineLimit(1...8)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .background(Color(nsColor: .textBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
                }
                .onSubmit {
                    guard !isGenerating else { return }
                    send()
                }
                .onAppear {
                    if ProcessInfo.processInfo.environment["MACHBOOST_UI_TESTING"] == "1" {
                        composerIsFocused = true
                    }
                }

            ZStack {
                Button {
                    stop()
                } label: {
                    Image(systemName: "stop.fill")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .accessibilityLabel("Stop generation")
                .help("Stop generation")
                .keyboardShortcut(.escape, modifiers: [])
                .opacity(isGenerating ? 1 : 0)
                .allowsHitTesting(isGenerating)
                .accessibilityHidden(!isGenerating)
                .zIndex(isGenerating ? 1 : 0)

                Button {
                    send()
                } label: {
                    Image(systemName: "arrow.up")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.borderedProminent)
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .accessibilityLabel("Send message")
                .help("Send")
                .keyboardShortcut(.return, modifiers: .command)
                .opacity(isGenerating ? 0 : 1)
                .allowsHitTesting(!isGenerating)
                .accessibilityHidden(isGenerating)
                .zIndex(isGenerating ? 0 : 1)
            }
            .frame(width: 52, height: 36)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .frame(maxWidth: 980)
        .frame(maxWidth: .infinity)
    }

    private var selectableModels: [CatalogModel] {
        appState.catalog
            .filter { $0.cached && $0.support == "ready" }
            .sorted { lhs, rhs in
                if lhs.recommended != rhs.recommended { return lhs.recommended }
                return lhs.displayName < rhs.displayName
            }
    }

    private var isGenerating: Bool { activeRequestID != nil }

    private var selectedModel: CatalogModel? {
        appState.model(named: conversation.model)
    }

    private var selectedWorkspace: WorkspaceSummary? {
        appState.workspace(id: conversation.workspaceID)
    }

    private var workspaceHelp: String {
        guard let workspace = selectedWorkspace else {
            return "Choose a repository for local code search"
        }
        let revision = workspace.revision.map { " at \($0)" } ?? ""
        return "\(workspace.fileCount) indexed files\(revision)"
    }

    private func selectAvailableModelIfNeeded() {
        guard
            !selectableModels.contains(where: {
                $0.name == conversation.model || $0.repository == conversation.model
            }),
            let model = selectableModels.first
        else {
            return
        }
        conversation.model = model.name
        conversation.updatedAt = .now
        try? modelContext.save()
    }

    private func chooseWorkspace() {
        let panel = NSOpenPanel()
        panel.title = "Choose a Repository"
        panel.prompt = "Index Repository"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }

        isAddingWorkspace = true
        Task {
            defer { isAddingWorkspace = false }
            guard let workspace = await appState.registerWorkspace(path: url.path) else {
                return
            }
            conversation.workspaceID = workspace.id
            conversation.updatedAt = .now
            try? modelContext.save()
        }
    }

    private func send(_ textOverride: String? = nil) {
        let text = (textOverride ?? draft).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isGenerating else { return }
        let images = conversation.orderedAttachments.filter { $0.kind == .image }
        if !images.isEmpty, appState.model(named: conversation.model)?.supportsVision != true {
            appState.presentedError = "\(conversation.model) cannot read images. Choose a vision model first."
            return
        }

        draft = ""
        let user = ChatMessage(role: .user, content: text, conversation: conversation)
        conversation.messages.append(user)
        if conversation.messages.filter({ $0.role == .user }).count == 1 {
            conversation.title = String(text.prefix(52))
        }
        conversation.updatedAt = .now

        let assistant = ChatMessage(
            role: .assistant,
            content: "",
            createdAt: Date().addingTimeInterval(0.001),
            conversation: conversation
        )
        conversation.messages.append(assistant)
        activeAssistant = assistant
        try? modelContext.save()

        let requestID = "chat-\(UUID().uuidString.lowercased())"
        activeRequestID = requestID
        let apiMessages = conversation.orderedMessages.compactMap { message -> APIChatMessage? in
            guard message.id != assistant.id else { return nil }
            let isLastUser = message.id == user.id
            return APIChatMessage(
                role: message.role.rawValue,
                content: message.content,
                images: isLastUser && !images.isEmpty
                    ? images.map(\.importedPath)
                    : nil
            )
        }
        let request = ChatRequest(
            requestID: requestID,
            model: conversation.model,
            messages: apiMessages,
            context: conversation.orderedAttachments
                .filter { $0.kind == .text }
                .map(\.importedPath),
            options: .init(
                maxTokens: maxTokens,
                temperature: temperature,
                affinityKey: conversation.workspaceID.map { "workspace:\($0)" }
                    ?? conversation.id.uuidString
            ),
            workspaceID: conversation.workspaceID,
            reasoningStrength: selectedModel?.supportsReasoning == true
                && reasoningStrength != "off"
                ? reasoningStrength
                : nil
        )
        generationTask = Task { @MainActor in
            var streamedToolCalls: [APIToolCall] = []
            do {
                for try await event in appState.api.streamChat(request) {
                    if let error = event.error {
                        throw MachBoostAPIError.stream(error)
                    }
                    if let content = event.message?.content, !content.isEmpty {
                        assistant.content += content
                    }
                    if let thinking = event.message?.thinking, !thinking.isEmpty {
                        assistant.reasoningContent = (assistant.reasoningContent ?? "") + thinking
                    }
                    if let calls = event.message?.toolCalls, !calls.isEmpty {
                        streamedToolCalls.append(contentsOf: calls)
                        if let data = try? JSONEncoder().encode(streamedToolCalls) {
                            assistant.toolCallsJSON = String(decoding: data, as: UTF8.self)
                        }
                    }
                    if event.done {
                        apply(event: event, to: assistant)
                    }
                }
                try? modelContext.save()
            } catch is CancellationError {
                assistant.wasCancelled = true
            } catch {
                if assistant.content.isEmpty {
                    modelContext.delete(assistant)
                }
                appState.presentedError = error.localizedDescription
            }
            activeRequestID = nil
            activeAssistant = nil
            generationTask = nil
            conversation.updatedAt = .now
            try? modelContext.save()
            await appState.refreshMetrics()
        }
    }

    private func stop() {
        guard let activeRequestID else { return }
        activeAssistant?.wasCancelled = true
        try? modelContext.save()
        Task { @MainActor in
            _ = try? await appState.api.cancel(requestID: activeRequestID)
            generationTask?.cancel()
        }
    }

    private func regenerateAction(
        for message: ChatMessage,
        in messages: [ChatMessage]
    ) -> (() -> Void)? {
        guard
            !isGenerating,
            message.role == .assistant,
            message.id == messages.last?.id,
            let lastUser = messages.last(where: { $0.role == .user })
        else {
            return nil
        }
        let text = lastUser.content
        let cutoff = lastUser.createdAt
        return { regenerate(text: text, cutoff: cutoff) }
    }

    private func regenerate(text: String, cutoff: Date) {
        let replacedMessages = conversation.messages.filter { $0.createdAt >= cutoff }
        conversation.messages.removeAll { $0.createdAt >= cutoff }
        for message in replacedMessages {
            modelContext.delete(message)
        }
        try? modelContext.save()
        send(text)
    }

    private func edit(_ message: ChatMessage) {
        guard message.role == .user, !isGenerating else { return }
        draft = message.content
        let cutoff = message.createdAt
        for candidate in conversation.messages where candidate.createdAt >= cutoff {
            modelContext.delete(candidate)
        }
        try? modelContext.save()
    }

    private func apply(event: ChatEvent, to message: ChatMessage) {
        message.wasCancelled = event.doneReason == "cancelled"
        message.durationSeconds = event.totalDuration.map { Double($0) / 1_000_000_000 }
        message.timeToFirstTokenSeconds = event.machboost?.timeToFirstTokenSeconds
        message.generatedTokens = event.evalCount ?? event.machboost?.stats?.generatedTokens
        if let count = message.generatedTokens, let duration = event.evalDuration, duration > 0 {
            message.tokensPerSecond = Double(count) / (Double(duration) / 1_000_000_000)
        } else if
            let count = message.generatedTokens,
            let duration = event.machboost?.stats?.generationSeconds,
            duration > 0
        {
            message.tokensPerSecond = Double(count) / duration
        }
    }

    private func importAttachments(_ result: Result<[URL], Error>) {
        do {
            let urls = try result.get()
            let imported = try AttachmentStore.importURLs(urls, conversation: conversation)
            conversation.attachments.append(contentsOf: imported)
            conversation.updatedAt = .now
            try modelContext.save()
        } catch {
            appState.presentedError = error.localizedDescription
        }
    }

    private func remove(_ attachment: ChatAttachment) {
        AttachmentStore.remove(attachment)
        modelContext.delete(attachment)
        try? modelContext.save()
    }

    @ViewBuilder
    private func capabilityIcons(for model: CatalogModel) -> some View {
        HStack(spacing: 7) {
            if model.supportsVision {
                Image(systemName: "eye")
                    .help("Vision")
            }
            if model.supportsReasoning {
                Image(systemName: "brain")
                    .help("Reasoning")
            }
            if model.supportsTools {
                Image(systemName: "wrench.and.screwdriver")
                    .help("Tool calling")
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }
}

private struct MessageRow: View {
    @Bindable var message: ChatMessage
    let showsReasoning: Bool
    let onEdit: () -> Void
    let onRegenerate: (() -> Void)?

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: message.role == .user ? "person.crop.circle.fill" : "bolt.fill")
                .foregroundStyle(message.role == .user ? Color.secondary : Color.teal)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(message.role == .user ? "You" : "MachBoost")
                        .font(.caption.weight(.semibold))
                    Spacer()
                    messageActions
                }
                if
                    message.content.isEmpty,
                    message.reasoningContent?.isEmpty != false,
                    message.toolCallsJSON?.isEmpty != false
                {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    if
                        showsReasoning,
                        let reasoning = message.reasoningContent,
                        !reasoning.isEmpty
                    {
                        DisclosureGroup("Reasoning") {
                            MessageContentView(content: reasoning)
                                .padding(.top, 6)
                                .foregroundStyle(.secondary)
                        }
                        .font(.callout)
                    }
                    if !message.content.isEmpty {
                        MessageContentView(content: message.content)
                    }
                    if !toolCalls.isEmpty {
                        toolCallList
                    }
                }
                if message.role == .assistant, hasStats {
                    stats
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 22)
        .padding(.vertical, 16)
        .frame(maxWidth: 980, alignment: .leading)
        .frame(maxWidth: .infinity)
        .background(
            message.role == .user
                ? Color(nsColor: .controlBackgroundColor).opacity(0.55)
                : Color.clear
        )
        .contextMenu {
            Button("Copy") { copyMessage() }
            if message.role == .user {
                Button("Edit and resend", action: onEdit)
            }
            if let onRegenerate {
                Button("Regenerate", action: onRegenerate)
            }
        }
    }

    private var messageActions: some View {
        HStack(spacing: 8) {
            Button(action: copyMessage) {
                Image(systemName: "doc.on.doc")
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(.borderless)
            .accessibilityLabel("Copy message")
            .help("Copy message")
            if message.role == .user {
                Button(action: onEdit) {
                    Image(systemName: "pencil")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.borderless)
                .accessibilityLabel("Edit and resend")
                .help("Edit and resend")
            }
            if let onRegenerate {
                Button(action: onRegenerate) {
                    Image(systemName: "arrow.clockwise")
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(.borderless)
                .accessibilityLabel("Regenerate response")
                .accessibilityIdentifier("regenerate-response")
                .help("Regenerate")
            }
        }
        .foregroundStyle(.secondary)
    }

    private var hasStats: Bool {
        message.tokensPerSecond != nil
            || message.timeToFirstTokenSeconds != nil
            || message.wasCancelled
    }

    private var stats: some View {
        HStack(spacing: 10) {
            if let rate = message.tokensPerSecond {
                Label("\(rate, specifier: "%.1f") tok/s", systemImage: "gauge.with.dots.needle.50percent")
            }
            if let ttft = message.timeToFirstTokenSeconds {
                Label("\(ttft, specifier: "%.2f")s TTFT", systemImage: "timer")
            }
            if let tokens = message.generatedTokens {
                Text("\(tokens) tokens")
            }
            if message.wasCancelled {
                Text("Stopped")
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private var toolCalls: [APIToolCall] {
        guard
            let json = message.toolCallsJSON,
            let data = json.data(using: .utf8)
        else {
            return []
        }
        return (try? JSONDecoder().decode([APIToolCall].self, from: data)) ?? []
    }

    private var toolCallList: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(toolCalls.enumerated()), id: \.offset) { _, call in
                VStack(alignment: .leading, spacing: 4) {
                    Label(call.function.name, systemImage: "wrench.and.screwdriver")
                        .font(.caption.weight(.semibold))
                    if let arguments = call.function.arguments {
                        Text(prettyJSON(arguments))
                            .font(.caption.monospaced())
                            .textSelection(.enabled)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            }
        }
    }

    private func prettyJSON(_ value: JSONValue) -> String {
        guard
            let data = try? JSONEncoder().encode(value),
            let object = try? JSONSerialization.jsonObject(with: data),
            let pretty = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys]
            )
        else {
            return ""
        }
        return String(decoding: pretty, as: UTF8.self)
    }

    private func copyMessage() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(message.content, forType: .string)
    }
}
