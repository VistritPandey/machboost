import AppKit
import SwiftUI

struct ModelSearchField: NSViewRepresentable {
    @Binding var text: String
    let placeholder: String
    let accessibilityIdentifier: String

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    func makeNSView(context: Context) -> NSSearchField {
        let field = NSSearchField()
        field.placeholderString = placeholder
        field.delegate = context.coordinator
        field.sendsSearchStringImmediately = true
        field.setAccessibilityIdentifier(accessibilityIdentifier)
        return field
    }

    func updateNSView(_ field: NSSearchField, context: Context) {
        context.coordinator.parent = self
        field.placeholderString = placeholder
        if field.stringValue != text {
            field.stringValue = text
        }
    }

    final class Coordinator: NSObject, NSSearchFieldDelegate {
        var parent: ModelSearchField

        init(parent: ModelSearchField) {
            self.parent = parent
        }

        func controlTextDidChange(_ notification: Notification) {
            guard let field = notification.object as? NSSearchField else { return }
            let next = field.stringValue
            guard next != parent.text else { return }
            parent.text = next
        }
    }
}

struct ModelsView: View {
    @Environment(AppState.self) private var appState
    @State private var search = ""
    @State private var advancedRepository = ""
    @State private var pendingDownload: CatalogModel?
    @State private var pendingRepository: String?
    @State private var pendingDeletion: CatalogModel?
    @State private var hubModels: [CatalogModel] = []
    @State private var isSearchingHub = false
    @State private var hubSearchTask: Task<Void, Never>?

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Models")
                    .font(.title2.weight(.semibold))
                Spacer()
                Button {
                    Task { await appState.refreshAll() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Refresh models")
            }
            .padding(18)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    ModelSearchField(
                        text: $search,
                        placeholder: "Search models",
                        accessibilityIdentifier: "models-page-search-field"
                    )
                    .frame(height: 26)

                    if !appState.downloads.isEmpty {
                        activeDownloadsSection
                    }

                    if !recommendedModels.isEmpty {
                        modelSection(title: "Recommended", models: recommendedModels)
                    }
                    modelSection(title: "Catalog", models: remainingModels)
                    if isSearchingHub {
                        HStack(spacing: 8) {
                            ProgressView()
                                .controlSize(.small)
                            Text("Searching Hugging Face")
                                .foregroundStyle(.secondary)
                        }
                    } else if !filteredHubModels.isEmpty {
                        modelSection(title: "Hugging Face", models: filteredHubModels)
                        Text("Live MLX results. MachBoost checks runtime compatibility before downloading weights.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    advancedSection
                }
                .padding(20)
                .frame(maxWidth: 920, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
        }
        .confirmationDialog(
            "Download model?",
            isPresented: Binding(
                get: { pendingDownload != nil || pendingRepository != nil },
                set: { if !$0 { pendingDownload = nil; pendingRepository = nil } }
            )
        ) {
            Button("Download") {
                let model = pendingDownload?.name ?? pendingRepository
                pendingDownload = nil
                pendingRepository = nil
                if let model {
                    Task { await appState.pull(model: model) }
                }
            }
            Button("Cancel", role: .cancel) {
                pendingDownload = nil
                pendingRepository = nil
            }
        } message: {
            if let model = pendingDownload {
                Text(downloadMessage(for: model))
            } else if let repository = pendingRepository {
                Text("MachBoost will verify \(repository) against the bundled MLX runtime before downloading its weights.")
            }
        }
        .confirmationDialog(
            "Delete downloaded model?",
            isPresented: Binding(
                get: { pendingDeletion != nil },
                set: { if !$0 { pendingDeletion = nil } }
            )
        ) {
            Button("Delete Model", role: .destructive) {
                let model = pendingDeletion?.name
                pendingDeletion = nil
                if let model {
                    Task { await appState.deleteModel(model) }
                }
            }
            Button("Cancel", role: .cancel) {
                pendingDeletion = nil
            }
        } message: {
            if let model = pendingDeletion {
                Text(deleteMessage(for: model))
            }
        }
        .onChange(of: search) { _, value in
            scheduleHubSearch(value)
        }
        .onDisappear {
            hubSearchTask?.cancel()
        }
    }

    private var activeDownloadsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Active Downloads")
                .font(.headline)
            ForEach(appState.downloads.keys.sorted(), id: \.self) { model in
                if let event = appState.downloads[model] {
                    ActiveDownloadRow(
                        model: model,
                        event: event,
                        onCancel: { Task { await appState.cancelPull(model: model) } }
                    )
                }
            }
        }
    }

    @ViewBuilder
    private func modelSection(title: String, models: [CatalogModel]) -> some View {
        if !models.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text(title)
                    .font(.headline)
                ForEach(models) { model in
                    ModelRow(
                        model: model,
                        loaded: appState.loadedModels.contains {
                            $0.model == model.repository || $0.model == model.name
                        },
                        loading: appState.loadingModels.contains(model.name),
                        deleting: appState.deletingModels.contains(model.name),
                        download: appState.downloads[model.name],
                        onDownload: { pendingDownload = model },
                        onCancel: { Task { await appState.cancelPull(model: model.name) } },
                        onLoad: {
                            Task { await appState.load(model: model.name) }
                        },
                        onUnload: {
                            Task { await appState.stop(model: model.name) }
                        },
                        onDelete: { pendingDeletion = model }
                    )
                }
            }
        }
    }

    private var advancedSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Advanced repository")
                .font(.headline)
            HStack {
                TextField("mlx-community/model-name", text: $advancedRepository)
                    .textFieldStyle(.roundedBorder)
                Button {
                    let repository = advancedRepository.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !repository.isEmpty else { return }
                    pendingRepository = repository
                } label: {
                    Label("Verify and download", systemImage: "checkmark.shield")
                }
                .disabled(advancedRepository.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            Text("MLX and MLX-VLM Hugging Face repositories are supported in this app release.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.top, 4)
    }

    private var filteredModels: [CatalogModel] {
        let query = search.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !query.isEmpty else { return appState.catalog }
        return appState.catalog.filter {
            $0.name.lowercased().contains(query)
                || $0.displayName.lowercased().contains(query)
                || ($0.repository?.lowercased().contains(query) ?? false)
        }
    }

    private var recommendedModels: [CatalogModel] {
        filteredModels.filter { $0.recommended && appState.downloads[$0.name] == nil }
    }

    private var remainingModels: [CatalogModel] {
        filteredModels.filter { !$0.recommended && appState.downloads[$0.name] == nil }
    }

    private var filteredHubModels: [CatalogModel] {
        let localNames = Set(appState.catalog.flatMap { [$0.name, $0.repository].compactMap { $0 } })
        return hubModels.filter {
            !localNames.contains($0.name)
                && !($0.repository.map(localNames.contains) ?? false)
                && appState.downloads[$0.name] == nil
        }
    }

    private func scheduleHubSearch(_ value: String) {
        hubSearchTask?.cancel()
        let query = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard query.count >= 2 else {
            hubModels = []
            isSearchingHub = false
            return
        }
        isSearchingHub = true
        hubSearchTask = Task {
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            let models = await appState.searchHubModels(query: query)
            guard !Task.isCancelled else { return }
            hubModels = models
            isSearchingHub = false
        }
    }

    private func downloadMessage(for model: CatalogModel) -> String {
        let source = model.backend == "ollama-mlx" ? "Ollama" : "Hugging Face"
        var pieces = ["Download \(model.displayName) through \(source)?"]
        if let size = model.downloadSizeGB {
            pieces.append("Estimated download: \(size.formatted(.number.precision(.fractionLength(1)))) GB.")
        }
        pieces.append(
            model.backend == "ollama-mlx"
                ? "Weights stay in your local Ollama model cache."
                : "Weights stay in your local Hugging Face cache."
        )
        return pieces.joined(separator: " ")
    }

    private func deleteMessage(for model: CatalogModel) -> String {
        var pieces = ["Delete \(model.displayName) from this Mac?"]
        if let size = model.diskSizeGB {
            pieces.append(
                "This will free about \(size.formatted(.number.precision(.fractionLength(2)))) GB."
            )
        }
        if appState.loadedModels.contains(where: { $0.model == model.repository || $0.model == model.name }) {
            pieces.append("The resident model will be unloaded first.")
        }
        pieces.append("You can download it again later.")
        return pieces.joined(separator: " ")
    }
}

private struct ModelRow: View {
    let model: CatalogModel
    let loaded: Bool
    let loading: Bool
    let deleting: Bool
    let download: PullEvent?
    let onDownload: () -> Void
    let onCancel: () -> Void
    let onLoad: () -> Void
    let onUnload: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: 14) {
            Image(
                systemName: model.supportsReasoning
                    ? "brain.fill"
                    : model.supportsVision ? "eye.fill" : "text.bubble.fill"
            )
                .font(.title3)
                .foregroundStyle(model.supportsReasoning ? Color.green : model.supportsVision ? Color.indigo : Color.teal)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(model.displayName)
                        .font(.body.weight(.medium))
                    if loaded {
                        Label("Loaded", systemImage: "memorychip.fill")
                            .foregroundStyle(.green)
                    } else if model.support == "unsupported" || model.support == "missing_runtime" {
                        Label("Unsupported", systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                    } else if !model.tested {
                        Label("Verify before download", systemImage: "checkmark.shield")
                            .foregroundStyle(.secondary)
                    } else if model.cached {
                        Label("Downloaded", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                }
                HStack(spacing: 10) {
                    Text(model.name)
                    Text(model.backend.uppercased())
                    ForEach(model.capabilities, id: \.self) { capability in
                        Text(capability.capitalized)
                    }
                    if let size = model.diskSizeGB {
                        Text("\(size.formatted(.number.precision(.fractionLength(2)))) GB on disk")
                    } else if let size = model.downloadSizeGB {
                        Text("~\(size.formatted(.number.precision(.fractionLength(1)))) GB download")
                    }
                    if let memory = model.minimumMemoryGB {
                        Text("\(Int(memory)) GB memory")
                    }
                    if let contextLength = model.contextLength {
                        Text("\(contextLength.formatted()) context")
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)

                if model.support != "ready", let reason = model.supportReason {
                    Text(reason)
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .lineLimit(2)
                }
                if let sourceRepository = model.sourceRepository {
                    Text(sourceRepository)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }

                if let download {
                    ModelDownloadProgress(event: download)
                }
            }

            Spacer()

            if loading || deleting {
                ProgressView()
                    .controlSize(.small)
                    .accessibilityLabel(
                        deleting ? "Deleting \(model.displayName)" : "Loading \(model.displayName)"
                    )
            } else if download != nil {
                Button(action: onCancel) {
                    Image(systemName: "xmark")
                }
                .accessibilityLabel("Cancel download for \(model.displayName)")
                .help("Cancel download")
            } else if loaded {
                HStack(spacing: 8) {
                    Button(action: onUnload) {
                        Image(systemName: "eject")
                    }
                    .accessibilityLabel("Unload \(model.displayName)")
                    .help("Unload model")
                    if model.cached {
                        deleteButton
                    }
                }
            } else if
                !model.cached,
                model.support != "unsupported",
                model.support != "missing_runtime"
            {
                Button(action: onDownload) {
                    Image(systemName: "arrow.down.circle")
                }
                .accessibilityLabel("Download \(model.displayName)")
                .accessibilityIdentifier("download-model-\(model.name)")
                .help("Download model")
            } else if model.cached {
                HStack(spacing: 8) {
                    if model.support == "ready" {
                        Button(action: onLoad) {
                            Image(systemName: "play.fill")
                        }
                        .accessibilityLabel("Load \(model.displayName)")
                        .accessibilityIdentifier("load-model-\(model.name)")
                        .help("Load and warm model")
                    }
                    deleteButton
                }
            }
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .stroke(Color(nsColor: .separatorColor).opacity(0.7), lineWidth: 1)
        }
    }

    private var deleteButton: some View {
        Button(action: onDelete) {
            Image(systemName: "trash")
        }
        .accessibilityLabel("Delete \(model.displayName)")
        .accessibilityIdentifier("delete-model-\(model.name)")
        .help("Delete downloaded model")
    }
}

private struct ActiveDownloadRow: View {
    let model: String
    let event: PullEvent
    let onCancel: () -> Void

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: "arrow.down.circle.fill")
                .font(.title3)
                .foregroundStyle(.green)
                .frame(width: 28)
            VStack(alignment: .leading, spacing: 5) {
                Text(model)
                    .font(.body.weight(.medium))
                    .textSelection(.enabled)
                ModelDownloadProgress(event: event)
            }
            Spacer()
            Button(action: onCancel) {
                Image(systemName: "xmark")
            }
            .accessibilityLabel("Cancel download for \(model)")
            .help("Cancel download")
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .stroke(Color(nsColor: .separatorColor).opacity(0.7), lineWidth: 1)
        }
    }
}

private struct ModelDownloadProgress: View {
    let event: PullEvent

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 8) {
                Text(event.file ?? event.status ?? "Preparing download")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer()
                if let detail = progressDetail {
                    Text(detail)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            HStack(spacing: 10) {
                if let filesCompleted = event.filesCompleted,
                   let filesTotal = event.filesTotal,
                   filesTotal > 0 {
                    Text("\(filesCompleted) of \(filesTotal) files")
                }
                if let speed = event.speedBytesPerSecond, speed > 0 {
                    Text("\(Self.bytes(speed))/s")
                }
                if let eta = event.etaSeconds, eta > 0 {
                    Text("about \(Self.duration(eta)) left")
                }
            }
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.secondary)
            if let completed = event.completed, let total = event.total, total > 0 {
                ProgressView(value: Double(completed), total: Double(total))
            } else {
                ProgressView()
                    .controlSize(.small)
            }
        }
        .accessibilityIdentifier("model-download-progress")
    }

    private var progressDetail: String? {
        guard let completed = event.completed, let total = event.total, total > 0 else {
            return nil
        }
        let percent = Int((Double(completed) / Double(total) * 100).rounded())
        guard event.unit == "bytes" else { return "\(percent)%" }
        let completedText = ByteCountFormatter.string(fromByteCount: completed, countStyle: .file)
        let totalText = ByteCountFormatter.string(fromByteCount: total, countStyle: .file)
        return "\(percent)%  \(completedText) / \(totalText)"
    }

    private static func bytes(_ value: Double) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .file)
    }

    private static func duration(_ seconds: Double) -> String {
        let formatter = DateComponentsFormatter()
        formatter.unitsStyle = .abbreviated
        formatter.allowedUnits = seconds >= 3600 ? [.hour, .minute] : [.minute, .second]
        formatter.maximumUnitCount = 2
        return formatter.string(from: seconds) ?? "a moment"
    }
}

struct ModelOnboardingView: View {
    @Environment(AppState.self) private var appState
    @State private var pendingDownload: CatalogModel?

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Choose your first model")
                        .font(.title2.weight(.semibold))
                    Text("Nothing downloads until you confirm it.")
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if appState.catalog.contains(where: \.cached) {
                    Button("Continue") {
                        appState.showOnboarding = false
                    }
                    .buttonStyle(.borderedProminent)
                } else if let recommendedModel {
                    Button {
                        pendingDownload = recommendedModel
                    } label: {
                        Label(
                            "Download \(recommendedModel.displayName)",
                            systemImage: "arrow.down.circle"
                        )
                    }
                    .accessibilityIdentifier("onboarding-download-model")
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding(20)
            Divider()
            ModelsView()
        }
        .frame(minWidth: 760, minHeight: 600)
        .interactiveDismissDisabled(!appState.catalog.contains(where: \.cached))
        .confirmationDialog(
            "Download model?",
            isPresented: Binding(
                get: { pendingDownload != nil },
                set: { if !$0 { pendingDownload = nil } }
            )
        ) {
            Button("Download") {
                let model = pendingDownload?.name
                pendingDownload = nil
                if let model {
                    Task { await appState.pull(model: model) }
                }
            }
            Button("Cancel", role: .cancel) {
                pendingDownload = nil
            }
        } message: {
            if let model = pendingDownload {
                Text("Download \(model.displayName) from Hugging Face? Weights stay in your local cache.")
            }
        }
    }

    private var recommendedModel: CatalogModel? {
        appState.catalog.first {
            $0.recommended && !$0.cached && $0.support == "ready"
        }
    }
}
