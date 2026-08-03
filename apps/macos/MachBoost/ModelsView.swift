import SwiftUI

struct ModelsView: View {
    @Environment(AppState.self) private var appState
    @State private var search = ""
    @State private var advancedRepository = ""
    @State private var pendingDownload: CatalogModel?
    @State private var pendingRepository: String?

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
                    TextField("Search models", text: $search)
                        .textFieldStyle(.roundedBorder)

                    if !recommendedModels.isEmpty {
                        modelSection(title: "Recommended", models: recommendedModels)
                    }
                    modelSection(title: "Catalog", models: remainingModels)
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
                        download: appState.downloads[model.name],
                        onDownload: { pendingDownload = model },
                        onCancel: { Task { await appState.cancelPull(model: model.name) } },
                        onLoad: {
                            Task { await appState.load(model: model.name) }
                        },
                        onUnload: {
                            Task { await appState.stop(model: model.name) }
                        }
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
        filteredModels.filter(\.recommended)
    }

    private var remainingModels: [CatalogModel] {
        filteredModels.filter { !$0.recommended }
    }

    private func downloadMessage(for model: CatalogModel) -> String {
        var pieces = ["Download \(model.displayName) from Hugging Face?"]
        if let size = model.downloadSizeGB {
            pieces.append("Estimated download: \(size.formatted(.number.precision(.fractionLength(1)))) GB.")
        }
        pieces.append("Weights stay in your local Hugging Face cache.")
        return pieces.joined(separator: " ")
    }
}

private struct ModelRow: View {
    let model: CatalogModel
    let loaded: Bool
    let loading: Bool
    let download: PullEvent?
    let onDownload: () -> Void
    let onCancel: () -> Void
    let onLoad: () -> Void
    let onUnload: () -> Void

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: model.supportsVision ? "eye.fill" : "text.bubble.fill")
                .font(.title3)
                .foregroundStyle(model.supportsVision ? Color.indigo : Color.teal)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(model.displayName)
                        .font(.body.weight(.medium))
                    if loaded {
                        Label("Loaded", systemImage: "memorychip.fill")
                            .foregroundStyle(.green)
                    } else if model.support != "ready" {
                        Label("Unsupported", systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
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
                }
                .font(.caption)
                .foregroundStyle(.secondary)

                if model.support != "ready", let reason = model.supportReason {
                    Text(reason)
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .lineLimit(2)
                }

                if let download {
                    downloadProgress(download)
                }
            }

            Spacer()

            if loading {
                ProgressView()
                    .controlSize(.small)
                    .accessibilityLabel("Loading \(model.displayName)")
            } else if download != nil {
                Button(action: onCancel) {
                    Image(systemName: "xmark")
                }
                .accessibilityLabel("Cancel download for \(model.displayName)")
                .help("Cancel download")
            } else if loaded {
                Button(action: onUnload) {
                    Image(systemName: "eject")
                }
                .accessibilityLabel("Unload \(model.displayName)")
                .help("Unload model")
            } else if !model.cached {
                Button(action: onDownload) {
                    Image(systemName: "arrow.down.circle")
                }
                .accessibilityLabel("Download \(model.displayName)")
                .accessibilityIdentifier("download-model-\(model.name)")
                .help("Download model")
            } else if model.support == "ready" {
                Button(action: onLoad) {
                    Image(systemName: "play.fill")
                }
                .accessibilityLabel("Load \(model.displayName)")
                .accessibilityIdentifier("load-model-\(model.name)")
                .help("Load and warm model")
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

    @ViewBuilder
    private func downloadProgress(_ event: PullEvent) -> some View {
        HStack(spacing: 8) {
            if let completed = event.completed, let total = event.total, total > 0 {
                ProgressView(value: Double(completed), total: Double(total))
                    .frame(maxWidth: 260)
            } else {
                ProgressView()
                    .controlSize(.small)
            }
            Text(event.file ?? event.status ?? "Downloading")
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
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
