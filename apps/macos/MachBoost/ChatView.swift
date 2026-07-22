import AppKit
import SwiftData
import SwiftUI
import UniformTypeIdentifiers

struct ChatView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.modelContext) private var modelContext
    @Bindable var conversation: Conversation

    @State private var draft = ""
    @State private var generationTask: Task<Void, Never>?
    @State private var activeRequestID: String?
    @State private var isImporting = false
    @State private var showsGenerationControls = false
    @AppStorage("machboost.chat.maxTokens") private var maxTokens = 512
    @AppStorage("machboost.chat.temperature") private var temperature = 0.2

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
            .frame(maxWidth: 320)

            if let model = appState.model(named: conversation.model) {
                Text(model.backend.uppercased())
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
            }

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
        }
        .formStyle(.grouped)
        .frame(width: 300)
        .padding(.vertical, 6)
    }

    private var messageList: some View {
        let messages: [ChatMessage] = conversation.orderedMessages

        return ScrollViewReader { (proxy: ScrollViewProxy) in
            ScrollView(.vertical, showsIndicators: true) {
                LazyVStack(alignment: .leading, spacing: 2) {
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
                                onEdit: { edit(message) },
                                onRegenerate: message.role == .assistant
                                    ? { regenerate() }
                                    : nil
                            )
                            .id(message.id)
                        }
                    }
                }
                .padding(.vertical, 12)
            }
            .onChange(of: messages.count, perform: { _ in
                if let last = messages.last {
                    withAnimation(.easeOut(duration: 0.18)) {
                        proxy.scrollTo(last.id, anchor: UnitPoint.bottom)
                    }
                }
            })
            .onChange(
                of: messages.last?.content,
                perform: { _ in
                    if let last = messages.last {
                        proxy.scrollTo(last.id, anchor: UnitPoint.bottom)
                    }
                }
            )
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
                .help("Stop generation")
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
        .padding(12)
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

    private func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
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
                affinityKey: conversation.id.uuidString
            )
        )
        generationTask = Task { @MainActor in
            do {
                for try await event in appState.api.streamChat(request) {
                    if let error = event.error {
                        throw MachBoostAPIError.stream(error)
                    }
                    if let content = event.message?.content, !content.isEmpty {
                        assistant.content += content
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
            generationTask = nil
            conversation.updatedAt = .now
            try? modelContext.save()
            await appState.refreshMetrics()
        }
    }

    private func stop() {
        guard let activeRequestID else { return }
        Task { @MainActor in
            _ = try? await appState.api.cancel(requestID: activeRequestID)
            generationTask?.cancel()
        }
    }

    private func regenerate() {
        guard let lastUser = conversation.orderedMessages.last(where: { $0.role == .user }) else {
            return
        }
        let text = lastUser.content
        let cutoff = lastUser.createdAt
        for message in conversation.messages where message.createdAt >= cutoff {
            modelContext.delete(message)
        }
        draft = text
        try? modelContext.save()
        send()
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
}

private struct MessageRow: View {
    @Bindable var message: ChatMessage
    let onEdit: () -> Void
    let onRegenerate: (() -> Void)?
    @State private var hovering = false

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
                    if hovering {
                        messageActions
                    }
                }
                if message.content.isEmpty {
                    ProgressView()
                        .controlSize(.small)
                } else {
                    MessageContentView(content: message.content)
                }
                if message.role == .assistant, hasStats {
                    stats
                }
            }
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
        .background(
            message.role == .user
                ? Color(nsColor: .controlBackgroundColor).opacity(0.55)
                : Color.clear
        )
        .onHover { hovering = $0 }
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
            }
            .accessibilityLabel("Copy message")
            .help("Copy message")
            if message.role == .user {
                Button(action: onEdit) {
                    Image(systemName: "pencil")
                }
                .accessibilityLabel("Edit and resend")
                .help("Edit and resend")
            }
            if let onRegenerate {
                Button(action: onRegenerate) {
                    Image(systemName: "arrow.clockwise")
                }
                .accessibilityLabel("Regenerate response")
                .help("Regenerate")
            }
        }
        .buttonStyle(.plain)
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

    private func copyMessage() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(message.content, forType: .string)
    }
}
