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
    @State private var showsModelBrowser = false
    @State private var modelSearch = ""
    @State private var pendingModelDownload: CatalogModel?
    @State private var pendingToolApproval: APIToolCall?
    @State private var toolApprovalContinuation: CheckedContinuation<Bool, Never>?
    @State private var isCompactingContext = false
    @FocusState private var composerIsFocused: Bool
    @AppStorage("machboost.chat.maxTokens") private var maxTokens = 512
    @AppStorage("machboost.chat.temperature") private var temperature = 0.2
    @AppStorage("machboost.chat.reasoningStrength") private var reasoningStrength = "medium"
    @AppStorage("machboost.chat.showReasoning") private var showReasoning = true
    @AppStorage("machboost.chat.autoSummarize") private var autoSummarize = true
    @AppStorage("machboost.chat.summaryThreshold") private var summaryThreshold = 90
    @AppStorage("machboost.chat.codingMode") private var codingMode = true

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
        .confirmationDialog(
            "Download model?",
            isPresented: Binding(
                get: { pendingModelDownload != nil },
                set: { if !$0 { pendingModelDownload = nil } }
            )
        ) {
            Button("Download") {
                guard let model = pendingModelDownload else { return }
                pendingModelDownload = nil
                Task { await appState.pull(model: model.name) }
            }
            Button("Cancel", role: .cancel) {
                pendingModelDownload = nil
            }
        } message: {
            if let model = pendingModelDownload {
                Text(downloadMessage(for: model))
            }
        }
        .confirmationDialog(
            "Allow repository change?",
            isPresented: Binding(
                get: { pendingToolApproval != nil },
                set: { if !$0 { resolveToolApproval(false) } }
            )
        ) {
            Button("Apply Change") { resolveToolApproval(true) }
            Button("Deny", role: .cancel) { resolveToolApproval(false) }
        } message: {
            if let pendingToolApproval {
                Text(CodingWorkspace.summary(of: pendingToolApproval))
            }
        }
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
            Button {
                showsModelBrowser.toggle()
            } label: {
                HStack(spacing: 9) {
                    Image(systemName: modelIcon(selectedModel))
                        .foregroundStyle(.green)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(selectedModel?.displayName ?? conversation.model)
                            .font(.body.weight(.medium))
                            .lineLimit(1)
                        if let selectedModel {
                            Text(modelSubtitle(selectedModel))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                    Spacer(minLength: 8)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 10)
            .frame(width: 330, height: 36)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
            }
            .accessibilityIdentifier("chat-model-picker")
            .popover(isPresented: $showsModelBrowser, arrowEdge: .bottom) {
                modelBrowser
            }

            workspaceMenu

            Button {
                codingMode.toggle()
            } label: {
                Image(systemName: codingMode ? "hammer.fill" : "hammer")
                    .foregroundStyle(codingMode ? Color.green : Color.secondary)
            }
            .buttonStyle(.plain)
            .disabled(selectedWorkspace == nil || selectedModel?.supportsTools != true)
            .accessibilityLabel("Coding mode")
            .help("Coding mode")

            Spacer()

            Label(
                appState.inferenceLabel,
                systemImage: appState.inferenceMode == .team ? "network" : "desktopcomputer"
            )
            .font(.caption)
            .foregroundStyle(appState.inferenceMode == .team ? Color.green : Color.secondary)

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

            if isCompactingContext {
                Label("Summarizing", systemImage: "text.append")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.green)
            } else if let activeRequestID {
                Text(activeRequestID.suffix(8))
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 16)
        .frame(height: 48)
    }

    private var modelBrowser: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Search models", text: $modelSearch)
                    .textFieldStyle(.plain)
                Button {
                    Task { await appState.refreshAll() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .help("Refresh model catalog")
            }
            .padding(12)

            Divider()

            if browsableModels.isEmpty {
                ContentUnavailableView.search(text: modelSearch)
                    .frame(maxWidth: .infinity, minHeight: 260)
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(browsableModels) { model in
                            modelBrowserRow(model)
                            Divider()
                                .padding(.leading, 54)
                        }
                    }
                }
            }

            Divider()
            HStack {
                Label(
                    "\(browsableModels.filter(\.cached).count) ready on \(appState.inferenceLabel)",
                    systemImage: "checkmark.circle"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                Spacer()
                Text("MLX native models")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.green)
            }
            .padding(.horizontal, 12)
            .frame(height: 38)
        }
        .frame(width: 500, height: 470)
    }

    private func modelBrowserRow(_ model: CatalogModel) -> some View {
        let loaded = appState.activeLoadedModels.contains {
            $0.model == model.name || $0.model == model.repository
        }
        let downloading = appState.downloads[model.name] != nil
        let selected = model.name == conversation.model || model.repository == conversation.model

        return Button {
            if model.cached {
                conversation.model = model.name
                conversation.updatedAt = .now
                try? modelContext.save()
                showsModelBrowser = false
            } else if appState.inferenceMode == .local, !downloading {
                pendingModelDownload = model
            }
        } label: {
            HStack(spacing: 12) {
                Image(systemName: modelIcon(model))
                    .font(.title3)
                    .foregroundStyle(model.supportsReasoning ? Color.green : Color.accentColor)
                    .frame(width: 30)

                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 7) {
                        Text(model.displayName)
                            .font(.body.weight(.medium))
                            .lineLimit(1)
                        if model.recommended {
                            Text("Recommended")
                                .font(.caption2.weight(.semibold))
                                .foregroundStyle(.green)
                        }
                    }
                    Text(modelSubtitle(model))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    HStack(spacing: 8) {
                        ForEach(model.capabilities, id: \.self) { capability in
                            Text(capability.capitalized)
                        }
                        if let contextLength = model.contextLength {
                            Text("\(contextLength.formatted()) ctx")
                        }
                    }
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                }

                Spacer(minLength: 8)

                if downloading {
                    ProgressView()
                        .controlSize(.small)
                } else if loaded {
                    Label("Loaded", systemImage: "memorychip.fill")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.green)
                } else if selected {
                    Image(systemName: "checkmark")
                        .foregroundStyle(.green)
                } else if model.cached {
                    Text("Ready")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Image(systemName: "arrow.down.circle")
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(model.displayName), \(model.cached ? "ready" : "download")")
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
        .accessibilityLabel("Repository picker")
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
            Divider()
            Toggle("Summarize older turns automatically", isOn: $autoSummarize)
            if autoSummarize {
                Stepper(value: $summaryThreshold, in: 70...95, step: 5) {
                    LabeledContent("Summarize at", value: "\(summaryThreshold)%")
                }
                LabeledContent(
                    "Estimated use",
                    value: contextUsageRatio.formatted(.percent.precision(.fractionLength(0)))
                )
            }
            Button {
                summarizeNow()
            } label: {
                Label("Summarize Now", systemImage: "text.append")
            }
            .disabled(isGenerating || compactionCandidates.isEmpty)
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
        if !conversation.orderedAttachments.isEmpty || conversation.contextSummary != nil {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 6) {
                    if let summaryUpdatedAt = conversation.summaryUpdatedAt {
                        Label(
                            "Context summarized \(summaryUpdatedAt.formatted(date: .omitted, time: .shortened))",
                            systemImage: "text.append"
                        )
                        .font(.caption)
                        .foregroundStyle(.green)
                        .padding(.horizontal, 8)
                        .frame(height: 28)
                        .background(Color.green.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                        .help("Older messages remain in chat history but the model receives this summary instead")
                    }
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

            Group {
                if isGenerating {
                    Button {
                        stop()
                    } label: {
                        Image(systemName: "stop.fill")
                            .frame(width: 28, height: 28)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                    .accessibilityLabel("Stop generation")
                    .accessibilityIdentifier("stop-generation")
                    .help("Stop generation")
                    .keyboardShortcut(.escape, modifiers: [])
                } else {
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
                }
            }
            .frame(width: 52, height: 36)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .frame(maxWidth: 980)
        .frame(maxWidth: .infinity)
    }

    private var selectableModels: [CatalogModel] {
        appState.activeCatalog
            .filter {
                $0.cached
                    && $0.support == "ready"
                    && ($0.backend.hasPrefix("mlx") || $0.backend == "dflash")
            }
            .sorted { lhs, rhs in
                if lhs.recommended != rhs.recommended { return lhs.recommended }
                return lhs.displayName < rhs.displayName
            }
    }

    private var browsableModels: [CatalogModel] {
        let query = modelSearch.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return appState.activeCatalog
            .filter { model in
                guard
                    model.support == "ready",
                    model.backend.hasPrefix("mlx") || model.backend == "dflash"
                else {
                    return false
                }
                return query.isEmpty
                    || model.name.lowercased().contains(query)
                    || model.displayName.lowercased().contains(query)
                    || (model.repository?.lowercased().contains(query) ?? false)
            }
            .sorted { lhs, rhs in
                if lhs.cached != rhs.cached { return lhs.cached }
                if lhs.recommended != rhs.recommended { return lhs.recommended }
                return lhs.displayName.localizedCaseInsensitiveCompare(rhs.displayName)
                    == .orderedAscending
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

    private func modelIcon(_ model: CatalogModel?) -> String {
        guard let model else { return "cpu" }
        if model.supportsReasoning { return "brain" }
        if model.supportsVision { return "eye" }
        if model.supportsTools { return "wrench.and.screwdriver" }
        return "text.bubble"
    }

    private func modelSubtitle(_ model: CatalogModel) -> String {
        var parts = [model.backend.uppercased()]
        if let memory = model.minimumMemoryGB {
            parts.append("\(Int(memory)) GB memory")
        } else if let size = model.diskSizeGB ?? model.downloadSizeGB {
            parts.append("\(size.formatted(.number.precision(.fractionLength(1)))) GB")
        }
        if !model.cached {
            parts.append("Download required")
        }
        return parts.joined(separator: " · ")
    }

    private func downloadMessage(for model: CatalogModel) -> String {
        var message = "Download \(model.displayName) from Hugging Face?"
        if let size = model.downloadSizeGB {
            message += " The estimated download is \(size.formatted(.number.precision(.fractionLength(1)))) GB."
        }
        message += " Weights remain in the local Hugging Face cache."
        return message
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

        let requestPrefix = "chat-\(UUID().uuidString.lowercased())"
        activeRequestID = requestPrefix
        let workspace = selectedWorkspace
        let codingActive = codingMode
            && workspace != nil
            && selectedModel?.supportsTools == true
        generationTask = Task { @MainActor in
            do {
                try? await appState.reportTeamPresence(
                    workspace: workspace,
                    model: conversation.model
                )
                let messages = try requestMessages(
                    currentUser: user,
                    excluding: assistant,
                    images: images,
                    codingActive: codingActive
                )
                try await runGenerationLoop(
                    requestPrefix: requestPrefix,
                    messages: messages,
                    assistant: assistant,
                    workspace: workspace,
                    codingActive: codingActive
                )
                try? modelContext.save()
                if autoSummarize {
                    await compactContextIfNeeded(force: false)
                }
            } catch is CancellationError {
                assistant.wasCancelled = true
            } catch {
                if assistant.content.isEmpty, assistant.toolCallsJSON == nil {
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
        resolveToolApproval(false)
        activeAssistant?.wasCancelled = true
        try? modelContext.save()
        Task { @MainActor in
            _ = try? await appState.inferenceAPI.cancel(requestID: activeRequestID)
            generationTask?.cancel()
        }
    }

    private func runGenerationLoop(
        requestPrefix: String,
        messages initialMessages: [APIChatMessage],
        assistant: ChatMessage,
        workspace: WorkspaceSummary?,
        codingActive: Bool
    ) async throws {
        var transcript = initialMessages
        var allToolCalls: [APIToolCall] = []
        for round in 0 ..< (codingActive ? CodingWorkspace.maximumToolRounds : 1) {
            try Task.checkCancellation()
            let requestID = round == 0 ? requestPrefix : "\(requestPrefix)-\(round)"
            activeRequestID = requestID
            let request = ChatRequest(
                requestID: requestID,
                model: conversation.model,
                messages: transcript,
                context: appState.inferenceMode == .local && !codingActive
                    ? conversation.orderedAttachments
                        .filter { $0.kind == .text }
                        .map(\.importedPath)
                    : [],
                options: .init(
                    maxTokens: maxTokens,
                    temperature: temperature,
                    affinityKey: workspace?.revision.map { "workspace:\($0)" }
                        ?? conversation.id.uuidString
                ),
                workspaceID: appState.inferenceMode == .local && !codingActive
                    ? conversation.workspaceID
                    : nil,
                reasoningStrength: selectedModel?.supportsReasoning == true
                    && reasoningStrength != "off"
                    ? reasoningStrength
                    : nil,
                tools: codingActive ? CodingWorkspace.tools : nil
            )
            var roundContent = ""
            var roundToolCalls: [APIToolCall] = []
            for try await event in appState.inferenceAPI.streamChat(request) {
                if let error = event.error { throw MachBoostAPIError.stream(error) }
                if let content = event.message?.content, !content.isEmpty {
                    roundContent += content
                    assistant.content += content
                }
                if let thinking = event.message?.thinking, !thinking.isEmpty {
                    assistant.reasoningContent = (assistant.reasoningContent ?? "") + thinking
                }
                if let calls = event.message?.toolCalls, !calls.isEmpty {
                    roundToolCalls.append(contentsOf: calls)
                }
                if event.done { apply(event: event, to: assistant) }
            }
            guard codingActive, !roundToolCalls.isEmpty, let workspace else { return }

            allToolCalls.append(contentsOf: roundToolCalls)
            if let data = try? JSONEncoder().encode(allToolCalls) {
                assistant.toolCallsJSON = String(decoding: data, as: UTF8.self)
            }
            transcript.append(
                APIChatMessage(
                    role: MessageRole.assistant.rawValue,
                    content: roundContent,
                    toolCalls: roundToolCalls
                )
            )
            for call in roundToolCalls {
                try Task.checkCancellation()
                let approved: Bool
                if CodingWorkspace.isMutating(call) {
                    approved = await requestToolApproval(call)
                } else {
                    approved = true
                }
                let result: String
                if !approved {
                    result = toolError("The user denied this repository change.")
                } else {
                    do {
                        result = try CodingWorkspace.execute(
                            call,
                            workspaceRoot: workspace.path
                        ).content
                    } catch {
                        result = toolError(error.localizedDescription)
                    }
                }
                transcript.append(
                    APIChatMessage(
                        role: "tool",
                        content: result,
                        toolName: call.function.name,
                        toolCallID: call.id
                    )
                )
            }
            if round == CodingWorkspace.maximumToolRounds - 1 {
                throw MachBoostAPIError.stream(
                    "Coding mode reached its tool-call limit. Start a new turn to continue."
                )
            }
        }
    }

    private func requestMessages(
        currentUser: ChatMessage,
        excluding assistant: ChatMessage,
        images: [ChatAttachment],
        codingActive: Bool
    ) throws -> [APIChatMessage] {
        var messages: [APIChatMessage] = []
        if codingActive {
            messages.append(
                APIChatMessage(role: MessageRole.system.rawValue, content: CodingWorkspace.systemPrompt)
            )
        }
        if let summary = conversation.contextSummary, !summary.isEmpty {
            messages.append(
                APIChatMessage(
                    role: MessageRole.system.rawValue,
                    content: "Conversation summary from earlier turns:\n\n\(summary)"
                )
            )
        }
        if appState.inferenceMode == .team,
           let attachmentContext = try remoteTextAttachmentContext() {
            messages.append(
                APIChatMessage(
                    role: MessageRole.system.rawValue,
                    content: attachmentContext
                )
            )
        }
        let imageReferences = try imageReferences(images)
        messages.append(contentsOf: conversation.orderedMessages.compactMap { message in
            guard
                message.id != assistant.id,
                conversation.summarizedThrough.map({ message.createdAt > $0 }) ?? true
            else {
                return nil
            }
            let isCurrentUser = message.id == currentUser.id
            return APIChatMessage(
                role: message.role.rawValue,
                content: message.content,
                images: isCurrentUser && !imageReferences.isEmpty ? imageReferences : nil
            )
        })
        return messages
    }

    private func requestToolApproval(_ call: APIToolCall) async -> Bool {
        await withCheckedContinuation { continuation in
            pendingToolApproval = call
            toolApprovalContinuation = continuation
        }
    }

    private func resolveToolApproval(_ approved: Bool) {
        guard let continuation = toolApprovalContinuation else {
            pendingToolApproval = nil
            return
        }
        toolApprovalContinuation = nil
        pendingToolApproval = nil
        continuation.resume(returning: approved)
    }

    private func imageReferences(_ images: [ChatAttachment]) throws -> [String] {
        guard appState.inferenceMode == .team else {
            return images.map(\.importedPath)
        }
        return try images.map { image in
            let data = try Data(contentsOf: URL(fileURLWithPath: image.importedPath))
            guard data.count <= 25 * 1024 * 1024 else {
                throw MachBoostAPIError.stream("\(image.displayName) exceeds the 25 MB image limit.")
            }
            let mimeType: String
            switch URL(fileURLWithPath: image.importedPath).pathExtension.lowercased() {
            case "jpg", "jpeg": mimeType = "image/jpeg"
            case "gif": mimeType = "image/gif"
            case "webp": mimeType = "image/webp"
            default: mimeType = "image/png"
            }
            return "data:\(mimeType);base64,\(data.base64EncodedString())"
        }
    }

    private func remoteTextAttachmentContext() throws -> String? {
        let attachments = conversation.orderedAttachments.filter { $0.kind == .text }
        guard !attachments.isEmpty else { return nil }
        var remaining = 128_000
        var sections: [String] = []
        for attachment in attachments where remaining > 0 {
            let data = try Data(contentsOf: URL(fileURLWithPath: attachment.importedPath))
            let chunk = data.prefix(min(data.count, min(64_000, remaining)))
            sections.append(
                "FILE: \(attachment.displayName)\n\(String(decoding: chunk, as: UTF8.self))"
            )
            remaining -= chunk.count
        }
        return "Explicitly attached local files:\n\n" + sections.joined(separator: "\n\n")
    }

    private func toolError(_ message: String) -> String {
        guard let data = try? JSONSerialization.data(
            withJSONObject: ["error": message],
            options: [.sortedKeys]
        ) else { return #"{"error":"tool failed"}"# }
        return String(decoding: data, as: UTF8.self)
    }

    private var contextUsageRatio: Double {
        let capacity = max(1, (selectedModel?.contextLength ?? 32_768) - maxTokens)
        return min(
            1,
            Double(
                ConversationCompaction.estimatedTokens(
                    summary: conversation.contextSummary,
                    messages: effectiveContextMessages
                )
            ) / Double(capacity)
        )
    }

    private var effectiveContextMessages: [ChatMessage] {
        conversation.orderedMessages.filter { message in
            conversation.summarizedThrough.map({ message.createdAt > $0 }) ?? true
        }
    }

    private var compactionCandidates: [ChatMessage] {
        ConversationCompaction.candidates(
            messages: effectiveContextMessages,
            keepRecent: 8
        )
    }

    private func summarizeNow() {
        guard !isGenerating, !compactionCandidates.isEmpty else { return }
        generationTask = Task { @MainActor in
            await compactContextIfNeeded(force: true)
            activeRequestID = nil
            generationTask = nil
        }
    }

    private func compactContextIfNeeded(force: Bool) async {
        let threshold = Double(summaryThreshold) / 100
        guard force || contextUsageRatio >= threshold else { return }
        let candidates = compactionCandidates
        guard let cutoff = candidates.last?.createdAt else { return }

        let requestID = "summary-\(UUID().uuidString.lowercased())"
        activeRequestID = requestID
        isCompactingContext = true
        defer { isCompactingContext = false }

        let transcript = candidates.map {
            "\($0.role.rawValue.uppercased()):\n\($0.content)"
        }.joined(separator: "\n\n")
        let prior = conversation.contextSummary.map {
            "Existing summary:\n\($0)\n\n"
        } ?? ""
        let request = ChatRequest(
            requestID: requestID,
            model: conversation.model,
            messages: [
                APIChatMessage(
                    role: MessageRole.system.rawValue,
                    content: """
                    Compress the supplied conversation into durable working context. Preserve decisions, constraints, file paths, APIs, errors, completed work, unresolved questions, and exact identifiers that future turns may need. Remove repetition and conversational filler. Return only the summary.
                    """
                ),
                APIChatMessage(
                    role: MessageRole.user.rawValue,
                    content: prior + transcript
                ),
            ],
            context: [],
            options: .init(
                maxTokens: min(1_024, maxTokens),
                temperature: 0,
                affinityKey: conversation.workspaceID.map { "workspace:\($0)" }
                    ?? conversation.id.uuidString
            ),
            reasoningStrength: nil
        )

        var summary = ""
        do {
            for try await event in appState.inferenceAPI.streamChat(request) {
                if let error = event.error { throw MachBoostAPIError.stream(error) }
                summary += event.message?.content ?? ""
            }
            summary = summary.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !summary.isEmpty else { return }
            conversation.contextSummary = summary
            conversation.summarizedThrough = cutoff
            conversation.summaryUpdatedAt = .now
            conversation.updatedAt = .now
            try? modelContext.save()
        } catch is CancellationError {
            return
        } catch {
            appState.presentedError = "The reply completed, but context summarization failed: \(error.localizedDescription)"
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

enum ConversationCompaction {
    static func estimatedTokens(summary: String?, messages: [ChatMessage]) -> Int {
        let summaryCharacters = summary?.count ?? 0
        let messageCharacters = messages.reduce(0) { total, message in
            total + message.content.count + 24
        }
        // Local chat tokenizers vary; three characters per token is intentionally
        // conservative so compaction runs before the backend must truncate.
        return Int(ceil(Double(summaryCharacters + messageCharacters) / 3))
    }

    static func candidates(
        messages: [ChatMessage],
        keepRecent: Int
    ) -> [ChatMessage] {
        let completed = messages.filter { !$0.content.isEmpty && !$0.wasCancelled }
        guard completed.count > keepRecent else { return [] }
        return Array(completed.dropLast(keepRecent))
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
                        .accessibilityIdentifier("message-reasoning")
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
                        .accessibilityIdentifier("tool-call-\(call.function.name)")
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
