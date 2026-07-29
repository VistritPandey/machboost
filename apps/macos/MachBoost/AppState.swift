import Foundation
import Observation

@MainActor
@Observable
final class AppState {
    private static let configurationKey = "machboost.server.configuration.v1"

    let daemon = DaemonManager()
    private let usesUITestAPI: Bool
    private(set) var configuration: ServerConfiguration
    private(set) var api: any MachBoostAPIProtocol
    private(set) var apiToken: String?
    private(set) var catalog: [CatalogModel] = []
    private(set) var loadedModels: [ModelInstance] = []
    private(set) var workspaces: [WorkspaceSummary] = []
    private(set) var metrics: ServerMetrics?
    private(set) var downloads: [String: PullEvent] = [:]
    private(set) var indexingWorkspaces: Set<String> = []
    private(set) var isRefreshing = false
    var showOnboarding = false
    var presentedError: String?

    init() {
        let configuration = Self.loadConfiguration()
        let token = KeychainStore.token()
        let usesUITestAPI = ProcessInfo.processInfo.environment["MACHBOOST_UI_TESTING"] == "1"
        self.usesUITestAPI = usesUITestAPI
        self.configuration = configuration
        self.apiToken = token
#if DEBUG
        if usesUITestAPI {
            let fixture = UITestMachBoostAPI()
            self.api = fixture
            self.catalog = fixture.catalogSnapshot()
        } else {
            self.api = MachBoostAPI(endpoint: configuration.endpoint, apiToken: token)
        }
#else
        self.api = MachBoostAPI(endpoint: configuration.endpoint, apiToken: token)
#endif
    }

    var serverIsRunning: Bool {
        usesUITestAPI || daemon.state == .running
    }

    func start() async {
        do {
            if configuration.lanEnabled {
                apiToken = try KeychainStore.tokenOrCreate()
            }
            rebuildAPI()
            try await daemon.start(configuration: configuration, apiToken: apiToken)
            await refreshAll()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func startUITestMode() async {
        guard usesUITestAPI else { return }
        await refreshAll()
    }

    func refreshAll() async {
        guard serverIsRunning else { return }
        isRefreshing = true
        defer { isRefreshing = false }
        do {
            async let catalog = api.catalog()
            async let models = api.models()
            async let metrics = api.metrics()
            async let workspaces = api.workspaces()
            let values = try await (catalog, models, metrics, workspaces)
            self.catalog = values.0
            self.loadedModels = values.1
            self.metrics = values.2
            self.workspaces = values.3
            if values.0.allSatisfy({ !$0.cached }) {
                showOnboarding = true
            }
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func refreshMetrics() async {
        guard serverIsRunning else { return }
        do {
            async let models = api.models()
            async let metrics = api.metrics()
            self.loadedModels = try await models
            self.metrics = try await metrics
        } catch {
            if daemon.state == .running {
                presentedError = error.localizedDescription
            }
        }
    }

    func applyConfiguration(_ updated: ServerConfiguration) async {
        do {
            var configuration = updated
            configuration.port = min(65_535, max(1_024, configuration.port))
            configuration.replicas = min(8, max(1, configuration.replicas))
            configuration.maxQueue = max(0, configuration.maxQueue)
            let previousConfiguration = self.configuration
            let previousToken = apiToken
            var nextToken = apiToken
            if configuration.lanEnabled {
                nextToken = try KeychainStore.tokenOrCreate()
            }
            try await daemon.restart(
                currentEndpoint: previousConfiguration.endpoint,
                currentAPIToken: previousToken,
                configuration: configuration,
                apiToken: nextToken
            )
            self.configuration = configuration
            apiToken = nextToken
            Self.saveConfiguration(configuration)
            rebuildAPI()
            await refreshAll()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func rotateToken() async {
        do {
            let previousToken = apiToken
            let nextToken = try KeychainStore.generateToken()
            if configuration.lanEnabled {
                try await daemon.restart(
                    currentEndpoint: configuration.endpoint,
                    currentAPIToken: previousToken,
                    configuration: configuration,
                    apiToken: nextToken
                )
            }
            apiToken = nextToken
            rebuildAPI()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func pull(model: String) async {
        let requestID = "pull-\(UUID().uuidString.lowercased())"
        do {
            let preflight = try await api.preflight(model: model)
            guard preflight.supported else {
                throw AppStateError.unsupportedModel(preflight.reason ?? "Unknown architecture")
            }
            for try await event in api.streamPull(model: model, requestID: requestID) {
                downloads[model] = event
                if let error = event.error {
                    throw MachBoostAPIError.stream(error)
                }
            }
            downloads.removeValue(forKey: model)
            await refreshAll()
        } catch {
            downloads.removeValue(forKey: model)
            presentedError = error.localizedDescription
        }
    }

    func cancelPull(model: String) async {
        guard let requestID = downloads[model]?.requestID else { return }
        _ = try? await api.cancel(requestID: requestID)
    }

    func stop(model: String) async {
        do {
            try await api.stop(model: model)
            await refreshMetrics()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func registerWorkspace(path: String) async -> WorkspaceSummary? {
        let operationID = "new:\(path)"
        indexingWorkspaces.insert(operationID)
        defer { indexingWorkspaces.remove(operationID) }
        do {
            let workspace = try await api.registerWorkspace(path: path, name: nil)
            workspaces.removeAll { $0.id == workspace.id }
            workspaces.insert(workspace, at: 0)
            return workspace
        } catch {
            presentedError = error.localizedDescription
            return nil
        }
    }

    func reindexWorkspace(id: String) async {
        indexingWorkspaces.insert(id)
        defer { indexingWorkspaces.remove(id) }
        do {
            let workspace = try await api.reindexWorkspace(id: id)
            workspaces.removeAll { $0.id == workspace.id }
            workspaces.insert(workspace, at: 0)
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func removeWorkspace(id: String) async {
        do {
            try await api.removeWorkspace(id: id)
            workspaces.removeAll { $0.id == id }
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func unloadAllModels() async {
        do {
            try await api.stop()
            await refreshMetrics()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func pauseServer() async {
        await daemon.shutdown(endpoint: configuration.endpoint, apiToken: apiToken)
        loadedModels = []
        metrics = nil
    }

    func resumeServer() async {
        await start()
    }

    func shutdown() async {
        await daemon.shutdown(endpoint: configuration.endpoint, apiToken: apiToken)
    }

    func model(named name: String) -> CatalogModel? {
        catalog.first { $0.name == name || $0.repository == name }
    }

    func workspace(id: String?) -> WorkspaceSummary? {
        guard let id else { return nil }
        return workspaces.first { $0.id == id }
    }

    private func rebuildAPI() {
#if DEBUG
        guard !usesUITestAPI else { return }
#endif
        api = MachBoostAPI(endpoint: configuration.endpoint, apiToken: apiToken)
    }

    private static func loadConfiguration() -> ServerConfiguration {
        guard
            let data = UserDefaults.standard.data(forKey: configurationKey),
            let configuration = try? JSONDecoder().decode(ServerConfiguration.self, from: data)
        else {
            return ServerConfiguration()
        }
        return configuration
    }

    private static func saveConfiguration(_ configuration: ServerConfiguration) {
        guard let data = try? JSONEncoder().encode(configuration) else { return }
        UserDefaults.standard.set(data, forKey: configurationKey)
    }
}

enum AppStateError: LocalizedError {
    case unsupportedModel(String)

    var errorDescription: String? {
        switch self {
        case let .unsupportedModel(reason):
            "This model is not compatible with the bundled MLX runtime: \(reason)"
        }
    }
}
