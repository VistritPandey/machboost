import AppKit
import SwiftData
import SwiftUI
import UniformTypeIdentifiers

@MainActor
private final class ToolApprovalCoordinator: ObservableObject {
    @Published private(set) var call: APIToolCall?
    private var continuation: CheckedContinuation<Bool, Never>?

    func request(_ call: APIToolCall) async -> Bool {
        await withCheckedContinuation { continuation in
            self.continuation = continuation
            self.call = call
        }
    }

    func resolve(_ approved: Bool) {
        guard let continuation else {
            call = nil
            return
        }
        self.continuation = nil
        call = nil
        continuation.resume(returning: approved)
    }
}

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
    @State private var modelFilter = ModelBrowserFilter.all
    @State private var pendingModelDownload: CatalogModel?
    @StateObject private var toolApproval = ToolApprovalCoordinator()
    @State private var pendingPermissionMode: CodingPermissionMode?
    @State private var isCompactingContext = false
    @State private var showsWorkspaceChanges = false
    @State private var workspaceChangeScope = WorkspaceChangeScope.conversation
    @State private var workspaceChanges = WorkspaceChangeSet.empty
    @State private var isRefreshingWorkspaceChanges = false
    @State private var workspaceChangesTask: Task<Void, Never>?
    @State private var isPreparingCodingPrefix = false
    @FocusState private var composerIsFocused: Bool
    @AppStorage("machboost.chat.maxTokens") private var maxTokens = 512
    @AppStorage("machboost.chat.temperature") private var temperature = 0.2
    @AppStorage("machboost.chat.reasoningStrength") private var reasoningStrength = "off"
    @AppStorage("machboost.chat.showReasoning") private var showReasoning = true
    @AppStorage("machboost.chat.autoSummarize") private var autoSummarize = true
    @AppStorage("machboost.chat.summaryThreshold") private var summaryThreshold = 90
    @AppStorage("machboost.chat.codingMode") private var codingMode = true
    @AppStorage("machboost.chat.permissionMode") private var permissionModeValue =
        CodingPermissionMode.automatic.rawValue
    @AppStorage("machboost.chat.routeMode") private var routeModeValue =
        ChatRouteMode.localOnly.rawValue
    @AppStorage("machboost.chat.providerID") private var providerID = ""
    @AppStorage("machboost.chat.providerModel") private var providerModel = ""

    var body: some View {
        responsiveChatSurface
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
                get: { toolApproval.call != nil },
                set: { if !$0 { resolveToolApproval(false) } }
            )
        ) {
            Button("Apply Change") { resolveToolApproval(true) }
            Button("Deny", role: .cancel) { resolveToolApproval(false) }
        } message: {
            if let call = toolApproval.call {
                Text(CodingWorkspace.summary(of: call))
            }
        }
        .confirmationDialog(
            "Bypass repository permissions?",
            isPresented: Binding(
                get: { pendingPermissionMode == .bypass },
                set: { if !$0 { pendingPermissionMode = nil } }
            )
        ) {
            Button("Enable Bypass", role: .destructive) {
                permissionModeValue = CodingPermissionMode.bypass.rawValue
                pendingPermissionMode = nil
            }
            Button("Cancel", role: .cancel) {
                pendingPermissionMode = nil
            }
        } message: {
            Text("MachBoost will approve every repository tool automatically. The model still cannot access files outside the selected repository.")
        }
        .onAppear {
            maxTokens = ConversationCompaction.clampedMaxTokens(maxTokens)
            summaryThreshold = ConversationCompaction.clampedThreshold(summaryThreshold)
            selectAvailableModelIfNeeded()
            selectAvailableProviderIfNeeded()
            activateSelectedInferenceHost()
#if DEBUG
            if let mode = ProcessInfo.processInfo.environment["MACHBOOST_UI_TEST_PERMISSION_MODE"],
               CodingPermissionMode(rawValue: mode) != nil {
                permissionModeValue = mode
            }
#endif
        }
        .onChange(of: selectableModels.map(\.name)) {
            selectAvailableModelIfNeeded()
        }
        .onChange(of: appState.providers.map { "\($0.id):\($0.models.joined(separator: ",")):\($0.hasSecret)" }) {
            selectAvailableProviderIfNeeded()
        }
        .onChange(of: conversation.workspaceID) {
            workspaceChangeScope = .conversation
            workspaceChanges = .empty
            if showsWorkspaceChanges {
                refreshWorkspaceChanges()
            }
        }
        .onChange(of: workspaceChangeScope) {
            if showsWorkspaceChanges {
                refreshWorkspaceChanges()
            }
        }
        .onDisappear {
            stop()
            workspaceChangesTask?.cancel()
        }
        .task(id: codingWarmupKey) {
            await prepareCodingPrefixIfNeeded()
        }
    }

    private var responsiveChatSurface: some View {
        GeometryReader { geometry in
            if showsWorkspaceChanges, geometry.size.width < 880 {
                workspaceChangesPanel
                    .transition(.move(edge: .trailing))
            } else {
                HStack(spacing: 0) {
                    chatSurface
                        .frame(minWidth: 0, maxWidth: .infinity)

                    if showsWorkspaceChanges {
                        Divider()
                        workspaceChangesPanel
                            .frame(
                                width: min(
                                    440,
                                    max(320, geometry.size.width * 0.32)
                                )
                            )
                            .transition(.move(edge: .trailing))
                    }
                }
            }
        }
        .animation(.easeInOut(duration: 0.18), value: showsWorkspaceChanges)
    }

    @ViewBuilder
    private var workspaceChangesPanel: some View {
        if let workspace = selectedWorkspace {
            WorkspaceChangesView(
                snapshot: workspaceChanges,
                workspaceRoot: workspace.path,
                isRefreshing: isRefreshingWorkspaceChanges,
                scope: $workspaceChangeScope,
                onRefresh: { refreshWorkspaceChanges() },
                onClose: { showsWorkspaceChanges = false }
            )
        } else {
            ContentUnavailableView(
                "No repository selected",
                systemImage: "folder.badge.questionmark"
            )
        }
    }

    private var chatSurface: some View {
        VStack(spacing: 0) {
            header
            Divider()
            messageList
            Divider()
            contextStrip
            composer
        }
    }

    private var header: some View {
        GeometryReader { geometry in
            let compact = geometry.size.width < 780
            let narrow = geometry.size.width < 590
            HStack(spacing: compact ? 8 : 12) {
                Button {
                    showsModelBrowser.toggle()
                } label: {
                    HStack(spacing: 9) {
                        Image(systemName: modelIcon(selectedModel))
                            .font(.body.weight(.semibold))
                            .foregroundStyle(.green)
                            .frame(width: 28, height: 28)
                            .background(Color.green.opacity(0.12))
                            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                        VStack(alignment: .leading, spacing: 1) {
                            Text(selectedModel?.displayName ?? conversation.model)
                                .font(.body.weight(.semibold))
                                .lineLimit(1)
                            if !compact, let selectedModel {
                                Text(modelSubtitle(selectedModel))
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }
                        }
                        Spacer(minLength: 4)
                        Image(systemName: "chevron.down")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 7)
                .frame(
                    width: narrow ? 190 : (compact ? 250 : min(330, geometry.size.width * 0.38)),
                    height: 40
                )
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
                }
                .accessibilityIdentifier("chat-model-picker")
                .popover(isPresented: $showsModelBrowser, arrowEdge: .bottom) {
                    modelBrowser
                }

                workspaceMenu(compact: narrow)

                if codingSessionAvailable {
                    Button {
                        showsWorkspaceChanges.toggle()
                        if showsWorkspaceChanges {
                            refreshWorkspaceChanges()
                        }
                    } label: {
                        Image(systemName: "sidebar.trailing")
                            .foregroundStyle(showsWorkspaceChanges ? Color.green : Color.secondary)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Workspace changes")
                    .accessibilityIdentifier("workspace-changes-toggle")
                    .help("Show workspace changes")
                }

                Spacer(minLength: 0)

                inferenceHostMenu(compact: compact)

                routeMenu(compact: compact)

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

                if !compact, isCompactingContext {
                    Label("Summarizing", systemImage: "text.append")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.green)
                } else if !compact, isPreparingCodingPrefix {
                    Label("Preparing Dev mode", systemImage: "bolt.horizontal.circle")
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.green)
                } else if !compact, let activeRequestID {
                    Text(activeRequestID.suffix(8))
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, compact ? 10 : 16)
        }
        .frame(height: 52)
    }

    private var modelBrowser: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Search models", text: $modelSearch)
                    .textFieldStyle(.plain)
                    .accessibilityLabel("Search models")
                    .accessibilityIdentifier("model-search-field")
                Button {
                    Task { await appState.refreshAll() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .help("Refresh model catalog")
            }
            .padding(12)

            Picker("Model filter", selection: $modelFilter) {
                ForEach(ModelBrowserFilter.allCases) { filter in
                    Label(filter.title, systemImage: filter.icon).tag(filter)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.horizontal, 12)
            .padding(.bottom, 10)

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
        let download = appState.downloads[model.name]
        let downloading = download != nil
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
                    .frame(width: 30, height: 30)
                    .background(Color.green.opacity(selected ? 0.18 : 0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))

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

                if let download {
                    VStack(alignment: .trailing, spacing: 4) {
                        if
                            let completed = download.completed,
                            let total = download.total,
                            total > 0
                        {
                            ProgressView(value: Double(completed), total: Double(total))
                                .frame(width: 90)
                            Text("\(Int((Double(completed) / Double(total) * 100).rounded()))%")
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                        } else {
                            ProgressView()
                                .controlSize(.small)
                            Text(download.status ?? "Preparing")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .accessibilityIdentifier("chat-model-download-progress")
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
            .background(selected ? Color.green.opacity(0.08) : Color.clear)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(model.displayName), \(model.cached ? "ready" : "download")")
    }

    private func workspaceMenu(compact: Bool) -> some View {
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
                if !compact {
                    Text(selectedWorkspace?.name ?? "Repository")
                        .lineLimit(1)
                    if let workspace = selectedWorkspace {
                        Text("\(workspace.fileCount)")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
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
            LabeledContent("Request route", value: routeMode.title)
            if routeMode.usesExternal {
                if appState.inferenceMode == .team {
                    LabeledContent("Paid API", value: "Managed by host")
                } else if let selectedProvider {
                    LabeledContent("Paid API", value: selectedProvider.name)
                    LabeledContent("API model", value: effectiveProviderModel)
                }
            }
            Divider()
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
                Picker("Reasoning", selection: reasoningSelection) {
                    if !selectedModelRequiresReasoning {
                        Text("Off").tag("off")
                    }
                    Text("Low").tag("low")
                    Text("Medium").tag("medium")
                    Text("High").tag("high")
                    Text("Max").tag("xhigh")
                }
                .pickerStyle(.segmented)
                .help(
                    selectedModelRequiresReasoning
                        ? "Muse Glimmer always reasons; Low is its fastest supported setting."
                        : "Control how much reasoning the model performs."
                )
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

    private func routeMenu(compact: Bool) -> some View {
        Menu {
            Section("Request route") {
                ForEach(ChatRouteMode.allCases) { mode in
                    Button {
                        routeModeValue = mode.rawValue
                        selectAvailableProviderIfNeeded()
                    } label: {
                        if mode == routeMode {
                            Label(mode.title, systemImage: "checkmark")
                        } else {
                            Label(mode.title, systemImage: mode.icon)
                        }
                    }
                    .disabled(mode.usesExternal && !externalRoutingAvailable)
                }
            }

            if routeMode.usesExternal {
                Divider()
                if appState.inferenceMode == .team {
                    Text("Paid API configured by host")
                } else if availableProviders.isEmpty {
                    Text("Add an API in Server")
                } else {
                    Section("Paid API model") {
                        ForEach(availableProviders) { provider in
                            Menu(provider.name) {
                                let models = provider.models.filter { $0 != "*" }
                                if models.isEmpty {
                                    providerModelButton(
                                        provider: provider,
                                        model: conversation.model,
                                        title: "Same model name"
                                    )
                                } else {
                                    ForEach(models, id: \.self) { model in
                                        providerModelButton(
                                            provider: provider,
                                            model: model,
                                            title: model
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } label: {
            if compact {
                Image(systemName: routeMode.icon)
                    .foregroundStyle(routeMode.usesExternal ? Color.green : Color.secondary)
            } else {
                Label(routeMode.shortTitle, systemImage: routeMode.icon)
                    .font(.caption)
                    .foregroundStyle(routeMode.usesExternal ? Color.green : Color.secondary)
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .accessibilityLabel("Inference route")
        .accessibilityValue(routeMode.title)
        .help(routeMode.help)
    }

    private func inferenceHostMenu(compact: Bool) -> some View {
        Menu {
            Section("Run this chat on") {
                ForEach(inferenceHostOptions) { option in
                    Button {
                        selectInferenceHost(option)
                    } label: {
                        if option.id == selectedInferenceHostID {
                            Label(option.name, systemImage: "checkmark")
                        } else {
                            Label(
                                option.name,
                                systemImage: option.id == InferenceHostOption.automaticID
                                    ? "arrow.triangle.branch"
                                    : "desktopcomputer"
                            )
                        }
                    }
                    .disabled(
                        option.id != InferenceHostOption.automaticID
                            && (!option.isOnline || option.detail == "Model unavailable")
                    )
                }
            }
            if let selected = selectedInferenceHostOption {
                Divider()
                Text(selected.detail)
            }
            Text("Falls back only if the preferred device fails before sending output.")
        } label: {
            if compact {
                Image(
                    systemName: selectedInferenceHostID == InferenceHostOption.automaticID
                        ? "arrow.triangle.branch"
                        : "desktopcomputer"
                )
                .foregroundStyle(.green)
            } else {
                Label(
                    selectedInferenceHostOption?.name ?? "Automatic",
                    systemImage: selectedInferenceHostID == InferenceHostOption.automaticID
                        ? "arrow.triangle.branch"
                        : "desktopcomputer"
                )
                .font(.caption)
                .foregroundStyle(.green)
                .lineLimit(1)
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .accessibilityLabel("Preferred inference device")
        .help("Choose the first device for this chat")
    }

    private func providerModelButton(
        provider: ProviderSummary,
        model: String,
        title: String
    ) -> some View {
        Button {
            providerID = provider.id
            providerModel = model
        } label: {
            if provider.id == providerID && model == effectiveProviderModel {
                Label(title, systemImage: "checkmark")
            } else {
                Text(title)
            }
        }
    }

    private var messageList: some View {
        let messages: [ChatMessage] = conversation.orderedMessages
        let scrollSignal = [
            messages.last?.content ?? "",
            messages.last?.reasoningContent ?? "",
            messages.last?.toolCallsJSON ?? "",
            messages.last?.toolActivityJSON ?? "",
        ].joined(separator: "|")

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
                                isStreaming: activeAssistant?.id == message.id
                                    && generationTask != nil,
                                workspaceRoot: selectedWorkspace?.path,
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
            .task(id: scrollSignal) {
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
        VStack(alignment: .leading, spacing: 8) {
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
                        guard !isGenerating, !isPreparingCodingPrefix else { return }
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
                        .disabled(
                            draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                || isPreparingCodingPrefix
                        )
                        .accessibilityLabel("Send message")
                        .help("Send")
                        .keyboardShortcut(.return, modifiers: .command)
                    }
                }
                .frame(width: 52, height: 36)
            }

            HStack(spacing: 8) {
                developerModeButton
                if codingSessionAvailable {
                    Divider()
                        .frame(height: 14)
                    permissionMenu
                }
                Spacer()
                if let selectedWorkspace {
                    Label(selectedWorkspace.name, systemImage: "folder")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
            .padding(.leading, 36)
            .padding(.trailing, 52)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .frame(maxWidth: 980)
        .frame(maxWidth: .infinity)
    }

    private var developerModeButton: some View {
        Button {
            guard selectedWorkspace != nil else {
                codingMode = true
                chooseWorkspace()
                return
            }
            guard selectedModel?.supportsTools == true || uiTestCodingFixtureEnabled else {
                appState.presentedError = "Choose a model with tool support to use Dev mode."
                return
            }
            codingMode.toggle()
            if !codingMode {
                showsWorkspaceChanges = false
            }
        } label: {
            Label("Dev mode", systemImage: "terminal")
                .font(.caption.weight(.semibold))
                .foregroundStyle(codingSessionAvailable ? Color.green : Color.secondary)
        }
        .buttonStyle(.borderless)
        .fixedSize()
        .accessibilityLabel("Developer mode")
        .accessibilityValue(codingSessionAvailable ? "On" : "Off")
        .help(
            selectedWorkspace == nil
                ? "Choose a repository and enable developer tools"
                : "Allow the model to inspect and edit the selected repository"
        )
    }

    private var permissionMenu: some View {
        Menu {
            ForEach(CodingPermissionMode.allCases) { mode in
                Button {
                    selectPermissionMode(mode)
                } label: {
                    if mode == permissionMode {
                        Label(mode.title, systemImage: "checkmark")
                    } else {
                        Label(mode.title, systemImage: mode.icon)
                    }
                }
            }
        } label: {
            Label(permissionMode.title, systemImage: permissionMode.icon)
                .font(.caption.weight(.semibold))
                .foregroundStyle(permissionMode == .bypass ? Color.orange : Color.green)
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .accessibilityLabel("Coding permission mode")
        .accessibilityValue(permissionMode.title)
        .accessibilityIdentifier("coding-permission-mode")
        .help("Choose how repository tool permissions are handled")
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
                let matchesSearch = query.isEmpty
                    || model.name.lowercased().contains(query)
                    || model.displayName.lowercased().contains(query)
                    || (model.repository?.lowercased().contains(query) ?? false)
                guard matchesSearch else { return false }
                switch modelFilter {
                case .all:
                    return true
                case .ready:
                    return model.cached
                case .loaded:
                    return appState.activeLoadedModels.contains {
                        $0.model == model.name || $0.model == model.repository
                    }
                case .vision:
                    return model.supportsVision
                case .tools:
                    return model.supportsTools
                }
            }
            .sorted { lhs, rhs in
                if lhs.cached != rhs.cached { return lhs.cached }
                if lhs.recommended != rhs.recommended { return lhs.recommended }
                return lhs.displayName.localizedCaseInsensitiveCompare(rhs.displayName)
                    == .orderedAscending
            }
    }

    private var isGenerating: Bool { activeRequestID != nil }

    private var selectedInferenceHostID: String {
        conversation.preferredInferenceHostID ?? InferenceHostOption.automaticID
    }

    private var inferenceHostOptions: [InferenceHostOption] {
        appState.inferenceHostOptions(for: conversation.model)
    }

    private var selectedInferenceHostOption: InferenceHostOption? {
        inferenceHostOptions.first { $0.id == selectedInferenceHostID }
            ?? inferenceHostOptions.first
    }

    private var codingWarmupKey: String {
        guard codingSessionAvailable else { return "off" }
        return [
            conversation.model,
            permissionMode.rawValue,
            selectedInferenceHostID,
            appState.inferenceMode.rawValue,
        ].joined(separator: "|")
    }

    private var conversationAffinityKey: String {
        "conversation:\(conversation.id.uuidString.lowercased())"
    }

    private var routeMode: ChatRouteMode {
        ChatRouteMode(rawValue: routeModeValue) ?? .localOnly
    }

    private var availableProviders: [ProviderSummary] {
        appState.providers.filter { $0.enabled && $0.hasSecret }
    }

    private var selectedProvider: ProviderSummary? {
        availableProviders.first { $0.id == providerID }
    }

    private var externalRoutingAvailable: Bool {
        appState.inferenceMode == .team || !availableProviders.isEmpty
    }

    private var effectiveProviderModel: String {
        guard let selectedProvider else { return providerModel }
        if selectedProvider.models.contains(providerModel) { return providerModel }
        return selectedProvider.models.first { $0 != "*" } ?? conversation.model
    }

    private func requestExtensions(memory: String? = nil) -> ChatRequest.Extensions {
        let mode = routeMode.usesExternal && !externalRoutingAvailable
            ? ChatRouteMode.localOnly
            : routeMode
        let remote = appState.inferenceMode == .team
        return .init(
            route: .init(
                mode: mode.rawValue,
                providerID: remote || !mode.usesExternal ? nil : selectedProvider?.id,
                model: remote || !mode.usesExternal ? nil : effectiveProviderModel
            ),
            memory: memory
        )
    }

    private var selectedModel: CatalogModel? {
        appState.model(named: conversation.model)
    }

    private var selectedModelRequiresReasoning: Bool {
        let identifiers = [
            conversation.model,
            selectedModel?.repository ?? "",
            selectedModel?.sourceRepository ?? "",
        ]
        return identifiers.contains { $0.lowercased().contains("muse-glimmer") }
    }

    private var reasoningSelection: Binding<String> {
        Binding(
            get: {
                selectedModelRequiresReasoning && reasoningStrength == "off"
                    ? "low"
                    : reasoningStrength
            },
            set: { reasoningStrength = $0 }
        )
    }

    private var effectiveReasoningStrength: String? {
        guard selectedModel?.supportsReasoning == true else { return nil }
        if selectedModelRequiresReasoning {
            return reasoningStrength == "off" ? "low" : reasoningStrength
        }
        return reasoningStrength == "off" ? nil : reasoningStrength
    }

    private var selectedWorkspace: WorkspaceSummary? {
        appState.workspace(id: conversation.workspaceID)
    }

    private var permissionMode: CodingPermissionMode {
        CodingPermissionMode(rawValue: permissionModeValue) ?? .automatic
    }

    private var codingSessionAvailable: Bool {
        codingMode
            && selectedWorkspace != nil
            && (selectedModel?.supportsTools == true || uiTestCodingFixtureEnabled)
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

    private func selectPermissionMode(_ mode: CodingPermissionMode) {
        if mode == .bypass, permissionMode != .bypass {
            pendingPermissionMode = mode
        } else {
            permissionModeValue = mode.rawValue
        }
    }

    private func selectInferenceHost(_ option: InferenceHostOption) {
        conversation.preferredInferenceHostID = option.id == InferenceHostOption.automaticID
            ? nil
            : option.id
        conversation.updatedAt = .now
        try? modelContext.save()
        if option.id == InferenceHostOption.localID {
            appState.useLocalInference()
        } else if let id = UUID(uuidString: option.id),
                  let profile = appState.teamHosts.first(where: { $0.id == id }) {
            appState.selectTeamHost(profile)
        } else if option.id == InferenceHostOption.automaticID,
                  !appState.teamHosts.isEmpty {
            Task { await appState.useTeamInference() }
        }
    }

    private func activateSelectedInferenceHost() {
        guard let option = selectedInferenceHostOption else { return }
        if option.id == InferenceHostOption.localID {
            appState.useLocalInference()
        } else if let id = UUID(uuidString: option.id),
                  let profile = appState.teamHosts.first(where: { $0.id == id }) {
            appState.selectTeamHost(profile)
        } else if option.id == InferenceHostOption.automaticID,
                  !appState.teamHosts.isEmpty {
            Task { await appState.useTeamInference() }
        }
    }

    private func prepareCodingPrefixIfNeeded() async {
        guard codingSessionAvailable, !isGenerating else { return }
        isPreparingCodingPrefix = true
        defer { isPreparingCodingPrefix = false }
        let requestID = "coding-prefix-\(UUID().uuidString.lowercased())"
        let request = ChatRequest(
            requestID: requestID,
            model: conversation.model,
            messages: [
                APIChatMessage(
                    role: MessageRole.system.rawValue,
                    content: CodingWorkspace.systemPrompt(for: permissionMode)
                ),
                APIChatMessage(role: MessageRole.user.rawValue, content: "Ready."),
            ],
            context: [],
            options: .init(
                maxTokens: 1,
                temperature: 0,
                affinityKey: conversationAffinityKey
            ),
            workspaceID: nil,
            workspaceTopK: nil,
            workspaceMaxChars: nil,
            reasoningStrength: effectiveReasoningStrength,
            tools: CodingWorkspace.tools(for: permissionMode),
            machboost: .init(
                route: .init(mode: ChatRouteMode.localOnly.rawValue),
                memory: "off"
            )
        )
        do {
            for try await _ in try appState.streamChat(
                request,
                preferredHostID: conversation.preferredInferenceHostID
            ) {}
            _ = appState.consumeInferenceRoute(requestID: requestID)
        } catch is CancellationError {
            return
        } catch {
            // The visible request reports actionable connection errors.
        }
    }

    private func selectAvailableProviderIfNeeded() {
        guard let provider = selectedProvider ?? availableProviders.first else { return }
        providerID = provider.id
        if !provider.models.contains(providerModel) {
            providerModel = provider.models.first { $0 != "*" } ?? conversation.model
        }
    }

    private func refreshWorkspaceChanges(show: Bool = false) {
        guard let workspace = selectedWorkspace else {
            workspaceChanges = .empty
            showsWorkspaceChanges = false
            return
        }
        if show {
            showsWorkspaceChanges = true
        }
        workspaceChangesTask?.cancel()
        isRefreshingWorkspaceChanges = true
        let root = workspace.path
        let scope = workspaceChangeScope
        let activities = conversationCodingActivities
        workspaceChangesTask = Task { @MainActor in
            let snapshot = await Task.detached(priority: .userInitiated) {
                switch scope {
                case .conversation:
                    WorkspaceChanges.session(
                        workspaceRoot: root,
                        activities: activities
                    )
                case .workingTree:
                    WorkspaceChanges.load(workspaceRoot: root)
                }
            }.value
            guard !Task.isCancelled else { return }
            workspaceChanges = snapshot
            isRefreshingWorkspaceChanges = false
        }
    }

    private var conversationCodingActivities: [CodingToolActivity] {
        conversation.orderedMessages.flatMap { message in
            guard
                let json = message.toolActivityJSON,
                let data = json.data(using: .utf8),
                let activities = try? JSONDecoder().decode(
                    [CodingToolActivity].self,
                    from: data
                )
            else {
                return [CodingToolActivity]()
            }
            return activities
        }
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
        guard !text.isEmpty, !isGenerating, !isPreparingCodingPrefix else { return }
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
            ?? (uiTestCodingFixtureEnabled ? appState.workspaces.first : nil)
        let codingActive = ((codingMode && codingSessionAvailable) || uiTestCodingFixtureEnabled)
            && workspace != nil
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
            } catch {
                if isCancellation(error) {
                    assistant.wasCancelled = true
                } else {
                    if assistant.content.isEmpty, assistant.toolCallsJSON == nil {
                        modelContext.delete(assistant)
                    }
                    appState.presentedError = error.localizedDescription
                }
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
            _ = await appState.cancelInference(requestID: activeRequestID)
            generationTask?.cancel()
        }
    }

    private func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        let message = error.localizedDescription
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return message == "cancelled"
            || message == "request cancelled"
            || message == "machboost cancelled"
    }

    private var uiTestCodingFixtureEnabled: Bool {
#if DEBUG
        ProcessInfo.processInfo.environment["MACHBOOST_UI_TEST_CODING"] == "1"
#else
        false
#endif
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
        var activities: [CodingToolActivity] = []
        var timeline: [AssistantTimelineEntry] = []
        var turnMetrics = GenerationTurnMetrics()
        var completedToolResults: [APIToolCall.Function: String] = [:]
        var forceFinalResponse = false
        defer {
            turnMetrics.apply(to: assistant)
        }
        let roundLimit = codingActive ? CodingWorkspace.maximumToolRounds + 1 : 1
        for round in 0 ..< roundLimit {
            try Task.checkCancellation()
            let requestID = round == 0 ? requestPrefix : "\(requestPrefix)-\(round)"
            activeRequestID = requestID
            let assistantContentPrefix = assistant.content
            let timelineStart = timeline.count
            var requestTranscript = transcript
            if forceFinalResponse {
                requestTranscript.append(
                    APIChatMessage(
                        role: MessageRole.system.rawValue,
                        content: "The repository tool phase is complete. Answer the user now using the results already returned. Do not request another tool."
                    )
                )
            }
            let request = ChatRequest(
                requestID: requestID,
                model: conversation.model,
                messages: requestTranscript,
                context: appState.inferenceMode == .local && !codingActive
                    ? conversation.orderedAttachments
                        .filter { $0.kind == .text }
                        .map(\.importedPath)
                    : [],
                options: .init(
                    maxTokens: maxTokens,
                    temperature: temperature,
                    affinityKey: conversationAffinityKey
                ),
                // Dev mode reads targeted files through tools; injecting the whole
                // repository map here duplicates context and delays the first token.
                workspaceID: appState.inferenceMode == .local && !codingActive
                    ? conversation.workspaceID
                    : nil,
                workspaceTopK: nil,
                workspaceMaxChars: nil,
                reasoningStrength: effectiveReasoningStrength,
                // Keep the tool schema stable across rounds so the resident
                // runtime can reuse the long coding prefix. The final-answer
                // instruction below is what prevents another tool request.
                tools: codingActive ? CodingWorkspace.tools(for: permissionMode) : nil,
                machboost: requestExtensions(memory: codingActive ? "off" : nil)
            )
            var roundContent = ""
            var roundToolCalls: [APIToolCall] = []
            var hasVisibleRoundContent = false
            for try await event in try appState.streamChat(
                request,
                preferredHostID: conversation.preferredInferenceHostID
            ) {
                if let error = event.error { throw MachBoostAPIError.stream(error) }
                if let thinking = event.message?.thinking, !thinking.isEmpty {
                    assistant.reasoningContent = (assistant.reasoningContent ?? "") + thinking
                    timeline.appendText(thinking, kind: .reasoning)
                    persist(timeline, to: assistant)
                }
                if let content = event.message?.content, !content.isEmpty {
                    let visibleChunk: String
                    if hasVisibleRoundContent {
                        visibleChunk = content
                    } else {
                        visibleChunk = String(content.drop(while: { $0.isWhitespace }))
                        hasVisibleRoundContent = !visibleChunk.isEmpty
                    }
                    if !visibleChunk.isEmpty {
                        roundContent += visibleChunk
                        assistant.content += visibleChunk
                        timeline.appendText(visibleChunk, kind: .content)
                        persist(timeline, to: assistant)
                    }
                }
                if let calls = event.message?.toolCalls, !calls.isEmpty {
                    roundToolCalls.append(contentsOf: calls)
                    allToolCalls.append(contentsOf: calls)
                    if let data = try? JSONEncoder().encode(allToolCalls) {
                        assistant.toolCallsJSON = String(decoding: data, as: UTF8.self)
                    }
                    let newActivities = calls.map { CodingToolActivity(call: $0) }
                    activities.append(contentsOf: newActivities)
                    timeline.append(
                        AssistantTimelineEntry(kind: .tools, activities: newActivities)
                    )
                    persist(activities, to: assistant)
                    persist(timeline, to: assistant)
                }
                if let fullContent = event.machboost?.fullContent {
                    reconcileRoundContent(
                        fullContent,
                        roundContent: &roundContent,
                        assistantContentPrefix: assistantContentPrefix,
                        assistant: assistant,
                        timeline: &timeline,
                        timelineStart: timelineStart
                    )
                }
                if event.done { turnMetrics.absorb(event) }
            }
            turnMetrics.recordRoute(appState.consumeInferenceRoute(requestID: requestID))
            if codingActive,
               !forceFinalResponse,
               roundToolCalls.isEmpty,
               roundContent.isEmpty,
               !activities.isEmpty {
                // Some tool-capable models end a round after reading a tool
                // result with only hidden reasoning/control tokens. Give them
                // one explicit chance to produce the user-facing answer.
                forceFinalResponse = true
                continue
            }
            guard
                codingActive,
                !forceFinalResponse,
                !roundToolCalls.isEmpty,
                let workspace
            else { return }

            transcript.append(
                APIChatMessage(
                    role: MessageRole.assistant.rawValue,
                    content: roundContent,
                    toolCalls: roundToolCalls
                )
            )
            let activityStart = activities.count - roundToolCalls.count
            var repeatedOnly = true
            for (offset, call) in roundToolCalls.enumerated() {
                try Task.checkCancellation()
                let activityIndex = activityStart + offset
                if let priorResult = completedToolResults[call.function] {
                    let reuseMessage = "This exact tool call already completed. Reusing its result instead of running it again."
                    activities[activityIndex].state = .succeeded
                    activities[activityIndex].output = reuseMessage
                    updateTimeline(&timeline, activity: activities[activityIndex])
                    persist(activities, to: assistant)
                    persist(timeline, to: assistant)
                    transcript.append(
                        APIChatMessage(
                            role: "tool",
                            content: priorResult,
                            toolName: call.function.name,
                            toolCallID: call.id
                        )
                    )
                    continue
                }
                repeatedOnly = false
                let permission = CodingWorkspace.permissionDecision(
                    for: call,
                    mode: permissionMode
                )
                let approved: Bool
                let denialMessage: String
                switch permission {
                case .allow:
                    approved = true
                    denialMessage = ""
                case .ask:
                    approved = await requestToolApproval(call)
                    denialMessage = "The user denied this repository change."
                case let .deny(reason):
                    approved = false
                    denialMessage = reason
                }
                let result: String
                if !approved {
                    result = toolError(denialMessage)
                    activities[activityIndex].state = .denied
                    activities[activityIndex].output = denialMessage
                    updateTimeline(&timeline, activity: activities[activityIndex])
                    persist(activities, to: assistant)
                    persist(timeline, to: assistant)
                } else {
                    activities[activityIndex].state = .running
                    updateTimeline(&timeline, activity: activities[activityIndex])
                    persist(activities, to: assistant)
                    persist(timeline, to: assistant)
                    do {
                        let toolResult = try await CodingWorkspace.execute(
                            call,
                            workspaceRoot: workspace.path
                        )
                        result = toolResult.content
                        activities[activityIndex].state = .succeeded
                        activities[activityIndex].output = toolResult.content
                        activities[activityIndex].changedPath = toolResult.changedPath
                        activities[activityIndex].changePatch = toolResult.changePatch
                    } catch {
                        result = toolError(error.localizedDescription)
                        activities[activityIndex].state = .failed
                        activities[activityIndex].output = error.localizedDescription
                    }
                    updateTimeline(&timeline, activity: activities[activityIndex])
                    persist(activities, to: assistant)
                    persist(timeline, to: assistant)
                    if activities[activityIndex].changedPath != nil {
                        refreshWorkspaceChanges()
                    }
                }
                completedToolResults[call.function] = result
                transcript.append(
                    APIChatMessage(
                        role: "tool",
                        content: result,
                        toolName: call.function.name,
                        toolCallID: call.id
                    )
                )
            }
            if repeatedOnly || round == CodingWorkspace.maximumToolRounds - 1 {
                forceFinalResponse = true
            }
        }
    }

    private func persist(_ activities: [CodingToolActivity], to message: ChatMessage) {
        guard let data = try? JSONEncoder().encode(activities) else { return }
        message.toolActivityJSON = String(decoding: data, as: UTF8.self)
        try? modelContext.save()
    }

    private func reconcileRoundContent(
        _ rawContent: String,
        roundContent: inout String,
        assistantContentPrefix: String,
        assistant: ChatMessage,
        timeline: inout [AssistantTimelineEntry],
        timelineStart: Int
    ) {
        let fullContent = CodingWorkspace.visibleAssistantText(rawContent)
        guard fullContent != roundContent else { return }
        assistant.content = assistantContentPrefix + fullContent
        let contentIndexes = timeline.indices.filter {
            $0 >= timelineStart && timeline[$0].kind == .content
        }
        if contentIndexes.isEmpty, !fullContent.isEmpty {
            timeline.append(AssistantTimelineEntry(kind: .content, text: fullContent))
        } else if contentIndexes.count == 1, let index = contentIndexes.first {
            timeline[index].text = fullContent
        } else {
            let streamed = contentIndexes.map { timeline[$0].text }.joined()
            if fullContent.hasPrefix(streamed), let index = contentIndexes.last {
                timeline[index].append(String(fullContent.dropFirst(streamed.count)))
            }
        }
        roundContent = fullContent
        persist(timeline, to: assistant)
    }

    private func persist(_ timeline: [AssistantTimelineEntry], to message: ChatMessage) {
        guard let data = try? JSONEncoder().encode(timeline) else { return }
        message.timelineJSON = String(decoding: data, as: UTF8.self)
    }

    private func updateTimeline(
        _ timeline: inout [AssistantTimelineEntry],
        activity: CodingToolActivity
    ) {
        for entryIndex in timeline.indices where timeline[entryIndex].kind == .tools {
            guard let activityIndex = timeline[entryIndex].activities.firstIndex(where: {
                $0.id == activity.id
            }) else { continue }
            timeline[entryIndex].activities[activityIndex] = activity
            return
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
                APIChatMessage(
                    role: MessageRole.system.rawValue,
                    content: CodingWorkspace.systemPrompt(for: permissionMode)
                )
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
            let content = message.role == .assistant
                ? CodingWorkspace.visibleAssistantText(message.content)
                : message.content
            if message.role == .assistant, content.isEmpty {
                return nil
            }
            return APIChatMessage(
                role: message.role.rawValue,
                content: content,
                images: isCurrentUser && !imageReferences.isEmpty ? imageReferences : nil
            )
        })
        return messages
    }

    private func requestToolApproval(_ call: APIToolCall) async -> Bool {
        await toolApproval.request(call)
    }

    private func resolveToolApproval(_ approved: Bool) {
        toolApproval.resolve(approved)
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
        let estimatedTokens = ConversationCompaction.estimatedTokens(
            summary: conversation.contextSummary,
            messages: effectiveContextMessages
        )
        guard force || ConversationCompaction.shouldCompact(
            estimatedTokens: estimatedTokens,
            contextLength: selectedModel?.contextLength ?? 32_768,
            reservedOutputTokens: maxTokens,
            thresholdPercent: summaryThreshold
        ) else { return }
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
                maxTokens: min(
                    1_024,
                    ConversationCompaction.clampedMaxTokens(maxTokens)
                ),
                temperature: 0,
                affinityKey: conversationAffinityKey
            ),
            reasoningStrength: selectedModelRequiresReasoning ? "low" : nil,
            machboost: requestExtensions()
        )

        var summary = ""
        do {
            for try await event in try appState.streamChat(
                request,
                preferredHostID: conversation.preferredInferenceHostID
            ) {
                if let error = event.error { throw MachBoostAPIError.stream(error) }
                summary += event.message?.content ?? ""
            }
            _ = appState.consumeInferenceRoute(requestID: requestID)
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
    static func clampedMaxTokens(_ value: Int) -> Int {
        min(4_096, max(32, value))
    }

    static func clampedThreshold(_ value: Int) -> Int {
        min(95, max(70, value))
    }

    static func estimatedTokens(summary: String?, messages: [ChatMessage]) -> Int {
        let summaryCharacters = summary?.count ?? 0
        let messageCharacters = messages.reduce(0) { total, message in
            total + message.content.count + 24
        }
        // Local chat tokenizers vary; three characters per token is intentionally
        // conservative so compaction runs before the backend must truncate.
        return Int(ceil(Double(summaryCharacters + messageCharacters) / 3))
    }

    static func shouldCompact(
        estimatedTokens: Int,
        contextLength: Int,
        reservedOutputTokens: Int,
        thresholdPercent: Int
    ) -> Bool {
        let capacity = max(1, contextLength - clampedMaxTokens(reservedOutputTokens))
        let threshold = Double(clampedThreshold(thresholdPercent)) / 100
        return Double(max(0, estimatedTokens)) / Double(capacity) >= threshold
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

private enum ModelBrowserFilter: String, CaseIterable, Identifiable {
    case all
    case ready
    case loaded
    case vision
    case tools

    var id: String { rawValue }

    var title: String {
        switch self {
        case .all: "All"
        case .ready: "Ready"
        case .loaded: "Loaded"
        case .vision: "Vision"
        case .tools: "Tools"
        }
    }

    var icon: String {
        switch self {
        case .all: "square.grid.2x2"
        case .ready: "checkmark.circle"
        case .loaded: "memorychip"
        case .vision: "eye"
        case .tools: "wrench.and.screwdriver"
        }
    }
}

private enum ChatRouteMode: String, CaseIterable, Identifiable {
    case localOnly = "local_only"
    case localFirst = "local_first"
    case externalFirst = "external_first"
    case externalOnly = "external_only"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .localOnly: "Local only"
        case .localFirst: "Local, then paid API"
        case .externalFirst: "Paid API, then local"
        case .externalOnly: "Paid API only"
        }
    }

    var shortTitle: String {
        switch self {
        case .localOnly: "Local"
        case .localFirst: "Local + API"
        case .externalFirst: "API + local"
        case .externalOnly: "API"
        }
    }

    var icon: String {
        switch self {
        case .localOnly: "desktopcomputer"
        case .localFirst: "arrow.right.circle"
        case .externalFirst: "arrow.left.arrow.right"
        case .externalOnly: "cloud"
        }
    }

    var usesExternal: Bool { self != .localOnly }

    var help: String {
        switch self {
        case .localOnly: "Always run on the selected MachBoost host"
        case .localFirst: "Use the paid API only when local inference is unavailable or overloaded"
        case .externalFirst: "Prefer the paid API and use local inference if it has a transient failure"
        case .externalOnly: "Always use the selected paid API"
        }
    }
}

private struct MessageRow: View {
    @Bindable var message: ChatMessage
    @State private var showsCodeChanges = false
    @State private var expandedToolIDs: Set<String> = []
    let showsReasoning: Bool
    let isStreaming: Bool
    let workspaceRoot: String?
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
                if hasNoVisibleOutput {
                    if isStreaming {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Text("No visible response was generated. Try regenerating the response.")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    if timeline.isEmpty {
                        legacyMessageBody
                    } else {
                        timelineBody
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
            || message.inferenceSource != nil
            || message.inferenceHostName != nil
            || message.wasCancelled
    }

    private var visibleContent: String {
        message.role == .assistant
            ? CodingWorkspace.visibleAssistantText(message.content)
            : message.content
    }

    @ViewBuilder
    private var legacyMessageBody: some View {
        if showsReasoning, let reasoning = message.reasoningContent, !reasoning.isEmpty {
            reasoningView(reasoning, isActive: isStreaming)
        }
        if !visibleContent.isEmpty {
            MessageContentView(content: visibleContent)
        }
        if !toolActivities.isEmpty {
            toolActivityList(toolActivities)
        }
    }

    private var timelineBody: some View {
        ForEach(timeline) { entry in
            Group {
                switch entry.kind {
                case .reasoning:
                    if showsReasoning, !entry.text.isEmpty {
                        reasoningView(
                            entry.text,
                            isActive: isStreaming && entry.id == timeline.last?.id
                        )
                    }
                case .content:
                    if !entry.text.isEmpty {
                        MessageContentView(
                            content: CodingWorkspace.visibleAssistantText(entry.text)
                        )
                    }
                case .tools:
                    if !entry.activities.isEmpty {
                        toolActivityList(entry.activities)
                    }
                }
            }
        }
    }

    private func reasoningView(_ reasoning: String, isActive: Bool) -> some View {
        StreamingReasoningDisclosure(reasoning: reasoning, isActive: isActive)
    }

    private var stats: some View {
        HStack(spacing: 10) {
            if let hostName = message.inferenceHostName {
                Label(
                    hostName,
                    systemImage: message.inferenceHostID == InferenceHostOption.localID
                        ? "desktopcomputer"
                        : "network"
                )
                .help("Inference ran on \(hostName)")
            }
            if let source = message.inferenceSource, source == "external" || source == "mixed" {
                Label(source == "external" ? "Paid API" : "Mixed route", systemImage: "cloud")
                    .help(message.providerID.map { "Provider: \($0)" } ?? "External provider")
            }
            if let rate = message.tokensPerSecond {
                Label(
                    "\(rate, specifier: "%.1f") \(message.inferenceSource == "external" ? "effective tok/s" : "tok/s")",
                    systemImage: "gauge.with.dots.needle.50percent"
                )
                .help(
                    message.inferenceSource == "external"
                        ? "Completion tokens divided by paid-provider request latency; upstream decode time was not reported"
                        : "Model tokens per decode second, including reasoning and tool protocol"
                )
            }
            if let ttft = message.timeToFirstTokenSeconds {
                Label("\(ttft, specifier: "%.2f")s TTFT", systemImage: "timer")
                    .help(ttftHelp)
            }
            if let cached = message.cachedPromptTokens,
               let prompt = message.promptTokens,
               cached > 0,
               prompt > 0 {
                Text("\(cached)/\(prompt) prefix cached")
                    .help("Prompt tokens reused without full prefill")
            }
            if let tokens = message.generatedTokens {
                Text("\(tokens) total model tokens")
                    .help("Includes answer, reasoning, and tool protocol tokens")
            }
            if let latency = message.providerLatencySeconds {
                Text("\(latency, specifier: "%.2f")s API")
            }
            if let cost = message.providerCostUSD, cost > 0 {
                Text(cost, format: .currency(code: "USD").precision(.fractionLength(4...6)))
                    .help("Estimated provider cost from configured token pricing")
            }
            if message.wasCancelled {
                Text("Stopped")
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private var ttftHelp: String {
        var parts: [String] = []
        if let load = message.modelLoadSeconds, load > 0.001 {
            parts.append("model load \(load.formatted(.number.precision(.fractionLength(2))))s")
        }
        if let queue = message.queueWaitSeconds, queue > 0.001 {
            parts.append("queue \(queue.formatted(.number.precision(.fractionLength(2))))s")
        }
        if let prefill = message.promptEvalSeconds, prefill > 0.001 {
            parts.append("prompt prefill \(prefill.formatted(.number.precision(.fractionLength(2))))s")
        }
        if let cached = message.cachedPromptTokens, cached > 0 {
            parts.append("\(cached) cached prompt tokens")
        }
        return parts.isEmpty
            ? "Time from request admission to the first model token"
            : "First-token breakdown: " + parts.joined(separator: ", ")
    }

    private var toolCalls: [APIToolCall] {
        guard
            let json = message.toolCallsJSON,
            let data = json.data(using: .utf8)
        else {
            return []
        }
        let decoded = (try? JSONDecoder().decode([APIToolCall].self, from: data)) ?? []
        return decoded.filter(CodingWorkspace.supports)
    }

    private var hasNoVisibleOutput: Bool {
        visibleContent.isEmpty
            && message.reasoningContent?.isEmpty != false
            && message.toolCallsJSON?.isEmpty != false
            && message.toolActivityJSON?.isEmpty != false
    }

    private var toolActivities: [CodingToolActivity] {
        if
            let json = message.toolActivityJSON,
            let data = json.data(using: .utf8),
            let activities = try? JSONDecoder().decode([CodingToolActivity].self, from: data)
        {
            return activities.filter { CodingWorkspace.supports($0.call) }
        }
        return toolCalls.map { CodingToolActivity(call: $0, state: .requested) }
    }

    private var timeline: [AssistantTimelineEntry] {
        guard
            let json = message.timelineJSON,
            let data = json.data(using: .utf8)
        else { return [] }
        return (try? JSONDecoder().decode([AssistantTimelineEntry].self, from: data)) ?? []
    }

    private func toolActivityList(_ activities: [CodingToolActivity]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(activities) { activity in
                VStack(alignment: .leading, spacing: 0) {
                    Button {
                        withAnimation(.easeInOut(duration: 0.14)) {
                            if expandedToolIDs.contains(activity.id) {
                                expandedToolIDs.remove(activity.id)
                            } else {
                                expandedToolIDs.insert(activity.id)
                            }
                        }
                    } label: {
                        HStack(spacing: 8) {
                            Image(
                                systemName: expandedToolIDs.contains(activity.id)
                                    ? "chevron.down"
                                    : "chevron.right"
                            )
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 12)
                            Image(systemName: statusIcon(activity.state))
                                .foregroundStyle(statusColor(activity.state))
                                .frame(width: 16)
                            Text(CodingWorkspace.activitySummary(of: activity.call))
                                .font(.callout.weight(.medium))
                                .lineLimit(1)
                            Spacer()
                            Text(statusLabel(activity.state))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    if expandedToolIDs.contains(activity.id) {
                        toolActivityDetails(activity)
                            .padding(.top, 8)
                            .padding(.leading, 20)
                    }
                }
                .accessibilityIdentifier("tool-call-\(activity.call.function.name)")
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            }
            if activities.contains(where: { $0.changedPath != nil && $0.changePatch != nil }) {
                codeChanges(activities)
            }
        }
    }

    @ViewBuilder
    private func toolActivityDetails(_ activity: CodingToolActivity) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(
                Array(CodingWorkspace.activityDetails(of: activity.call).enumerated()),
                id: \.offset
            ) { _, detail in
                LabeledContent(detail.0, value: detail.1)
                    .font(.caption)
            }
            if let output = activity.output, !output.isEmpty {
                Text(activity.state == .failed || activity.state == .denied ? "Message" : "Result")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                ScrollView([.horizontal, .vertical]) {
                    Text(CodingWorkspace.displayResult(output))
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 220)
            }
        }
    }

    private func codeChanges(_ activities: [CodingToolActivity]) -> some View {
        let changedActivities = activities.filter {
            $0.changedPath != nil && $0.changePatch != nil
        }
        return VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.easeInOut(duration: 0.16)) {
                    showsCodeChanges.toggle()
                }
            } label: {
                HStack {
                    Label(
                        "Code changes (\(changedActivities.count))",
                        systemImage: "doc.badge.gearshape"
                    )
                    Spacer()
                    Image(systemName: showsCodeChanges ? "chevron.down" : "chevron.right")
                        .font(.caption.weight(.semibold))
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .font(.callout.weight(.semibold))
            .foregroundStyle(.green)
            .accessibilityIdentifier("code-changes")

            if showsCodeChanges {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(changedActivities) { activity in
                        if let path = activity.changedPath, let patch = activity.changePatch {
                            VStack(alignment: .leading, spacing: 8) {
                                HStack {
                                    Label(path, systemImage: "doc.text")
                                        .font(.caption.weight(.semibold))
                                    Spacer()
                                    if CodingWorkspace.fileURL(
                                        relativePath: path,
                                        workspaceRoot: workspaceRoot
                                    ) != nil {
                                        Button {
                                            openFile(path)
                                        } label: {
                                            Label("Open File", systemImage: "arrow.up.forward.app")
                                        }
                                        .buttonStyle(.borderless)
                                        Button {
                                            revealFile(path)
                                        } label: {
                                            Label("Reveal", systemImage: "folder")
                                        }
                                        .buttonStyle(.borderless)
                                    }
                                }
                                ScrollView([.horizontal, .vertical]) {
                                    Text(patch)
                                        .font(.caption.monospaced())
                                        .textSelection(.enabled)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .accessibilityIdentifier("change-patch-\(activity.id)")
                                }
                                .frame(maxHeight: 280)
                                .padding(8)
                                .background(Color(nsColor: .textBackgroundColor))
                                .clipShape(RoundedRectangle(cornerRadius: 5, style: .continuous))
                            }
                        }
                    }
                }
                .padding(.top, 4)
            }
        }
        .padding(10)
        .background(Color.green.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
    }

    private func statusIcon(_ state: CodingToolState) -> String {
        switch state {
        case .requested: "wrench.and.screwdriver"
        case .queued: "clock"
        case .running: "arrow.trianglehead.2.clockwise.rotate.90"
        case .succeeded: "checkmark.circle.fill"
        case .denied: "hand.raised.fill"
        case .failed: "xmark.circle.fill"
        }
    }

    private func statusColor(_ state: CodingToolState) -> Color {
        switch state {
        case .requested, .queued, .running: .secondary
        case .succeeded: .green
        case .denied: .orange
        case .failed: .red
        }
    }

    private func statusLabel(_ state: CodingToolState) -> String {
        switch state {
        case .requested: "Requested"
        case .queued: "Queued"
        case .running: "Running"
        case .succeeded: "Done"
        case .denied: "Denied"
        case .failed: "Failed"
        }
    }

    private func openFile(_ path: String) {
        guard let url = CodingWorkspace.fileURL(relativePath: path, workspaceRoot: workspaceRoot) else {
            return
        }
        NSWorkspace.shared.open(url)
    }

    private func revealFile(_ path: String) {
        guard let url = CodingWorkspace.fileURL(relativePath: path, workspaceRoot: workspaceRoot) else {
            return
        }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    private func copyMessage() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(visibleContent, forType: .string)
    }
}

private struct StreamingReasoningDisclosure: View {
    let reasoning: String
    let isActive: Bool
    @State private var isExpanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeInOut(duration: 0.14)) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 7) {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 12)
                    if isActive {
                        ProgressView()
                            .controlSize(.mini)
                    }
                    Text(isActive ? "Reasoning…" : "Reasoning")
                    Spacer()
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .frame(maxWidth: .infinity, alignment: .leading)
            if isExpanded {
                MessageContentView(content: reasoning)
                    .padding(.top, 6)
                    .padding(.leading, 19)
                    .foregroundStyle(.secondary)
            }
        }
        .font(.callout)
        .accessibilityIdentifier("message-reasoning")
        .onAppear {
            isExpanded = isActive
        }
        .onChange(of: isActive) {
            withAnimation(.easeInOut(duration: 0.14)) {
                isExpanded = isActive
            }
        }
    }
}
