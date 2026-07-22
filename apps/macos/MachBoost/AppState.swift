import Foundation
import Observation

@MainActor
@Observable
final class AppState {
    private static let configurationKey = "machboost.server.configuration.v1"

    let daemon = DaemonManager()
    private(set) var configuration: ServerConfiguration
    private(set) var api: MachBoostAPI
    private(set) var apiToken: String?
    private(set) var catalog: [CatalogModel] = []
    private(set) var loadedModels: [ModelInstance] = []
    private(set) var metrics: ServerMetrics?
    private(set) var downloads: [String: PullEvent] = [:]
    private(set) var isRefreshing = false
    var showOnboarding = false
    var presentedError: String?

    init() {
        let configuration = Self.loadConfiguration()
        let token = KeychainStore.token()
        self.configuration = configuration
        self.apiToken = token
        self.api = MachBoostAPI(endpoint: configuration.endpoint, apiToken: token)
    }

    var serverIsRunning: Bool {
        daemon.state == .running
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

    func refreshAll() async {
        guard daemon.state == .running else { return }
        isRefreshing = true
        defer { isRefreshing = false }
        do {
            async let catalog = api.catalog()
            async let models = api.models()
            async let metrics = api.metrics()
            let values = try await (catalog, models, metrics)
            self.catalog = values.0
            self.loadedModels = values.1
            self.metrics = values.2
            if values.0.allSatisfy({ !$0.cached }) {
                showOnboarding = true
            }
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func refreshMetrics() async {
        guard daemon.state == .running else { return }
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
            if configuration.lanEnabled {
                apiToken = try KeychainStore.tokenOrCreate()
            }
            self.configuration = configuration
            Self.saveConfiguration(configuration)
            rebuildAPI()
            try await daemon.restart(
                configuration: configuration,
                apiToken: apiToken
            )
            await refreshAll()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func rotateToken() async {
        do {
            apiToken = try KeychainStore.generateToken()
            rebuildAPI()
            if configuration.lanEnabled {
                try await daemon.restart(
                    configuration: configuration,
                    apiToken: apiToken
                )
            }
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

    private func rebuildAPI() {
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
