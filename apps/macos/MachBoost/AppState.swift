import Foundation
import Observation

@MainActor
@Observable
final class AppState {
    private static let configurationKey = "machboost.server.configuration.v1"
    private static let teamProfileKey = "machboost.team.profile.v1"
    private static let teamProfilesKey = "machboost.team.profiles.v2"
    private static let inferenceModeKey = "machboost.inference.mode.v1"
    private static let deviceIDKey = "machboost.device.id.v1"
    private static let includeLocalPoolKey = "machboost.team.include-local.v1"
    private static let localPoolID = UUID(uuidString: "00000000-0000-0000-0000-000000000000")!

    let daemon = DaemonManager()
    let hostDiscovery = MachBoostHostDiscovery()
    private let usesUITestAPI: Bool
    private(set) var configuration: ServerConfiguration
    private(set) var api: any MachBoostAPIProtocol
    private(set) var inferenceAPI: any MachBoostAPIProtocol
    private(set) var apiToken: String?
    private(set) var catalog: [CatalogModel] = []
    private(set) var loadedModels: [ModelInstance] = []
    private(set) var teamCatalog: [CatalogModel] = []
    private(set) var teamLoadedModels: [ModelInstance] = []
    private(set) var teamHost: TeamHostProfile?
    private(set) var teamHosts: [TeamHostProfile] = []
    private(set) var teamHostSnapshots: [UUID: TeamHostSnapshot] = [:]
    private(set) var teamClient: TeamClient?
    private(set) var teamClients: [TeamClient] = []
    private(set) var teamModelRequests: [TeamModelRequest] = []
    private(set) var inferenceMode: InferenceMode
    private(set) var workspaces: [WorkspaceSummary] = []
    private(set) var metrics: ServerMetrics?
    private(set) var teamStatus: TeamStatus?
    private(set) var teamKeys: [TeamKey] = []
    private(set) var traces: [TraceSummary] = []
    private(set) var evaluations: [TraceEvaluation] = []
    private(set) var memories: [MemorySummary] = []
    private(set) var cacheMetrics: CacheMetrics?
    private(set) var providers: [ProviderSummary] = []
    private(set) var lastCreatedTeamToken: String?
    private(set) var downloads: [String: PullEvent] = [:]
    private(set) var loadingModels: Set<String> = []
    private(set) var lastModelLoad: ModelLoadResponse?
    private(set) var indexingWorkspaces: Set<String> = []
    private(set) var isRefreshing = false
    private var heartbeatTask: Task<Void, Never>?
    private var teamAPIs: [UUID: any MachBoostAPIProtocol] = [:]
    private var requestAPIs: [String: any MachBoostAPIProtocol] = [:]
    private var requestHostIDs: [String: UUID] = [:]
    private var reservedRequests: [UUID: Int] = [:]
    private var lastPresenceAt: [UUID: Date] = [:]
    let deviceID: String
    var showOnboarding = false
    var presentedError: String?

    init() {
        let environment = ProcessInfo.processInfo.environment
        let configuration = Self.loadConfiguration()
        let usesUITestAPI = environment["MACHBOOST_UI_TESTING"] == "1"
        let isTesting = usesUITestAPI
            || environment["MACHBOOST_TESTING"] == "1"
            || environment["XCTestConfigurationFilePath"] != nil
        let storedProfile = isTesting ? nil : Self.loadTeamProfile()
        let storedProfiles = isTesting ? [] : Self.loadTeamProfiles(fallback: storedProfile)
        let selectedProfile = storedProfile ?? storedProfiles.first
        let storedMode = InferenceMode(
            rawValue: UserDefaults.standard.string(forKey: Self.inferenceModeKey) ?? "local"
        ) ?? .local
        let deviceID = Self.loadDeviceID()
        self.usesUITestAPI = usesUITestAPI
        self.configuration = configuration
        self.apiToken = nil
        self.deviceID = deviceID
        self.teamHost = selectedProfile
        self.teamHosts = storedProfiles
        self.inferenceMode = selectedProfile == nil ? .local : storedMode
#if DEBUG
        if usesUITestAPI {
            let fixture = UITestMachBoostAPI()
            self.api = fixture
            self.inferenceAPI = fixture
            self.catalog = fixture.catalogSnapshot()
        } else {
            let local = MachBoostAPI(endpoint: configuration.endpoint, apiToken: nil)
            self.api = local
            self.inferenceAPI = local
        }
#else
        let local = MachBoostAPI(endpoint: configuration.endpoint, apiToken: nil)
        self.api = local
        self.inferenceAPI = local
#endif
    }

    var serverIsRunning: Bool {
        usesUITestAPI || daemon.state == .running
    }

    var activeCatalog: [CatalogModel] {
        inferenceMode == .team ? teamCatalog : catalog
    }

    var activeLoadedModels: [ModelInstance] {
        inferenceMode == .team ? teamLoadedModels : loadedModels
    }

    var inferenceLabel: String {
        if inferenceMode == .team, teamHosts.count > 1 {
            return "Host pool (\(teamHosts.count))"
        }
        if inferenceMode == .team, let teamHost {
            return teamHost.hostName
        }
        return "This Mac"
    }

    var teamIsConnected: Bool {
        inferenceMode == .team && teamHost != nil && !teamAPIs.isEmpty
    }

    var includeLocalInHostPool: Bool {
        get { UserDefaults.standard.object(forKey: Self.includeLocalPoolKey) as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: Self.includeLocalPoolKey) }
    }

    func connectToTeamHost(endpoint rawEndpoint: String, token: String) async {
        do {
            let endpoint = try Self.normalizedTeamEndpoint(rawEndpoint)
            let normalizedToken = token.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !normalizedToken.isEmpty else {
                throw AppStateError.invalidTeamHost("Enter the API key created by the host.")
            }
            let profileID = teamHosts.first(where: { $0.endpoint == endpoint })?.id ?? UUID()
            let remote = MachBoostAPI(
                endpoint: endpoint,
                apiToken: normalizedToken,
                deviceID: deviceID
            )
            let connected = try await remote.teamConnect()
            let profile = TeamHostProfile(
                id: profileID,
                endpoint: endpoint,
                hostName: connected.host.name,
                hostVersion: connected.host.version,
                principalName: connected.principal.name,
                connectedAt: .now
            )
            try await KeychainStore.saveTeamTokenAsync(
                normalizedToken,
                profileID: profileID
            )
            teamHost = profile
            teamHosts.removeAll { $0.id == profile.id }
            teamHosts.append(profile)
            teamHosts.sort { $0.hostName.localizedCaseInsensitiveCompare($1.hostName) == .orderedAscending }
            teamAPIs[profile.id] = remote
            teamHostSnapshots[profile.id] = TeamHostSnapshot(
                profile: profile,
                catalog: connected.models,
                loadedModels: connected.loadedModels,
                metrics: try? await remote.metrics(),
                isOnline: true,
                lastError: nil,
                updatedAt: .now
            )
            teamCatalog = connected.models
            teamLoadedModels = connected.loadedModels
            inferenceMode = .team
            inferenceAPI = remote
            Self.saveTeamProfile(profile)
            Self.saveTeamProfiles(teamHosts)
            Self.saveInferenceMode(.team)
            try await reportTeamPresence(workspace: nil, model: teamCatalog.first?.name)
            startHeartbeat()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func useLocalInference() {
        heartbeatTask?.cancel()
        heartbeatTask = nil
        inferenceMode = .local
        inferenceAPI = api
        Self.saveInferenceMode(.local)
    }

    func useTeamInference() async {
        guard teamHost != nil else {
            presentedError = "Connect to a Team host first."
            return
        }
        inferenceMode = .team
        Self.saveInferenceMode(.team)
        await reconnectTeamHosts()
    }

    func forgetTeamHost() {
        if let teamHost {
            Task {
                try? await KeychainStore.deleteTeamTokenAsync(profileID: teamHost.id)
            }
            teamHosts.removeAll { $0.id == teamHost.id }
            teamAPIs.removeValue(forKey: teamHost.id)
            teamHostSnapshots.removeValue(forKey: teamHost.id)
        }
        heartbeatTask?.cancel()
        heartbeatTask = nil
        teamHost = teamHosts.first
        teamClient = nil
        teamCatalog = []
        teamLoadedModels = []
        Self.saveTeamProfiles(teamHosts)
        if let next = teamHost {
            Self.saveTeamProfile(next)
            Task { await reconnectTeamHosts() }
        } else {
            inferenceMode = .local
            inferenceAPI = api
            UserDefaults.standard.removeObject(forKey: Self.teamProfileKey)
            Self.saveInferenceMode(.local)
        }
    }

    func reportTeamPresence(workspace: WorkspaceSummary?, model: String?) async throws {
        guard inferenceMode == .team else { return }
        teamClient = try await inferenceAPI.reportTeamPresence(
            deviceID: deviceID,
            deviceName: Host.current().localizedName ?? ProcessInfo.processInfo.hostName,
            appVersion: Bundle.main.object(
                forInfoDictionaryKey: "CFBundleShortVersionString"
            ) as? String ?? "development",
            workspaceName: workspace?.name,
            workspaceFingerprint: workspace?.revision,
            model: model
        )
    }

    func requestModelFromHost(_ model: String, note: String?) async {
        do {
            _ = try await inferenceAPI.requestTeamModel(
                model: model,
                deviceID: deviceID,
                note: note
            )
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func resolveModelRequest(
        _ request: TeamModelRequest,
        status: String,
        note: String? = nil
    ) async {
        do {
            _ = try await api.resolveTeamModelRequest(
                id: request.id,
                status: status,
                note: note
            )
            await refreshTeam()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func start() async {
        do {
            if configuration.lanEnabled {
                apiToken = try await KeychainStore.tokenOrCreateAsync()
            }
            rebuildAPI()
            try await daemon.start(configuration: configuration, apiToken: apiToken)
            hostDiscovery.start()
            updateHostAdvertisement()
            await refreshAll()
            if inferenceMode == .team {
                await reconnectTeamHosts()
            }
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
            await refreshTeam()
            await refreshMemoryAndProviders()
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

    func refreshTeam() async {
        guard serverIsRunning else { return }
        do {
            async let status = api.teamStatus()
            async let keys = api.teamKeys()
            async let traces = api.traces(limit: 100)
            async let evaluations = api.evaluations(limit: 25)
            async let clients = api.teamClients()
            async let modelRequests = api.teamModelRequests(status: nil)
            let values = try await (
                status, keys, traces, evaluations, clients, modelRequests
            )
            teamStatus = values.0
            teamKeys = values.1
            self.traces = values.2
            self.evaluations = values.3
            teamClients = values.4
            teamModelRequests = values.5
        } catch MachBoostAPIError.server(status: 404, message: _) {
            teamStatus = nil
            teamKeys = []
            traces = []
            evaluations = []
            teamClients = []
            teamModelRequests = []
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func refreshMemoryAndProviders() async {
        guard serverIsRunning else { return }
        do {
            async let memories = api.memories(workspaceID: nil)
            async let metrics = api.cacheMetrics()
            async let providers = api.providers()
            let values = try await (memories, metrics, providers)
            self.memories = values.0
            cacheMetrics = values.1
            self.providers = values.2
            if !usesUITestAPI {
                for provider in values.2 where !provider.hasSecret {
                    guard let secret = await KeychainStore.providerSecretAsync(id: provider.id)
                    else { continue }
                    try await api.setProviderSecret(id: provider.id, apiKey: secret)
                }
            }
        } catch MachBoostAPIError.server(status: 404, message: _) {
            memories = []
            cacheMetrics = nil
            providers = []
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func deleteMemory(id: String) async {
        do {
            try await api.deleteMemory(id: id)
            await refreshMemoryAndProviders()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func configureProvider(
        id: String?,
        name: String,
        baseURL: String,
        models: [String],
        apiKey: String,
        monthlyBudgetUSD: Double?
    ) async {
        do {
            let provider = try await api.configureProvider(
                id: id,
                name: name,
                baseURL: baseURL,
                models: models,
                apiKey: apiKey,
                monthlyBudgetUSD: monthlyBudgetUSD
            )
            if !apiKey.isEmpty {
                try await KeychainStore.saveProviderSecretAsync(apiKey, id: provider.id)
            }
            await refreshMemoryAndProviders()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func deleteProvider(id: String) async {
        do {
            try await api.deleteProvider(id: id)
            try await KeychainStore.deleteProviderSecretAsync(id: id)
            await refreshMemoryAndProviders()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func createTeamKey(
        name: String,
        allowedModels: [String],
        maxConcurrent: Int,
        requestsPerMinute: Int
    ) async {
        do {
            let created = try await api.createTeamKey(
                name: name,
                scopes: [
                    "inference",
                    "models:read",
                    "workspaces:read",
                    "traces:read",
                    "evaluations:read",
                    "evaluations:write",
                ],
                allowedModels: allowedModels,
                maxConcurrent: maxConcurrent,
                requestsPerMinute: requestsPerMinute
            )
            lastCreatedTeamToken = created.token
            await refreshTeam()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func clearCreatedTeamToken() {
        lastCreatedTeamToken = nil
    }

    func revokeTeamKey(id: String) async {
        do {
            try await api.revokeTeamKey(id: id)
            await refreshTeam()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func updateTeamSettings(
        traceMode: String,
        retentionDays: Int?,
        maxStorageBytes: Int64
    ) async {
        do {
            _ = try await api.updateTeamSettings(
                traceMode: traceMode,
                retentionDays: retentionDays,
                maxStorageBytes: maxStorageBytes
            )
            await refreshTeam()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func evaluateTraces(ids: [String], model: String?) async {
        guard !ids.isEmpty else { return }
        do {
            _ = try await api.evaluate(
                traceIDs: ids,
                name: "Team gateway evaluation",
                model: model
            )
            await refreshTeam()
        } catch {
            presentedError = error.localizedDescription
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
                nextToken = try await KeychainStore.tokenOrCreateAsync()
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
            updateHostAdvertisement()
            await refreshAll()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func rotateToken() async {
        do {
            let previousToken = apiToken
            let nextToken = try await KeychainStore.generateTokenAsync()
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

    func load(
        model: String,
        keepAlive: String = "forever",
        warmup: Bool = true
    ) async {
        guard serverIsRunning else {
            presentedError = AppStateError.serverNotRunning.localizedDescription
            return
        }
        loadingModels.insert(model)
        defer { loadingModels.remove(model) }
        do {
            lastModelLoad = try await api.load(
                model: model,
                keepAlive: keepAlive,
                warmup: warmup
            )
            await refreshMetrics()
        } catch {
            presentedError = error.localizedDescription
        }
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
        heartbeatTask?.cancel()
        await daemon.shutdown(endpoint: configuration.endpoint, apiToken: apiToken)
    }

    func model(named name: String) -> CatalogModel? {
        activeCatalog.first { $0.name == name || $0.repository == name }
    }

    func workspace(id: String?) -> WorkspaceSummary? {
        guard let id else { return nil }
        return workspaces.first { $0.id == id }
    }

    func streamChat(_ request: ChatRequest) throws -> AsyncThrowingStream<ChatEvent, Error> {
        let selected = try inferenceSelection(for: request.model)
        requestAPIs[request.requestID] = selected.api
        requestHostIDs[request.requestID] = selected.hostID
        reservedRequests[selected.hostID, default: 0] += 1
        let source = selected.api.streamChat(request)
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    for try await event in source {
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
                _ = await MainActor.run {
                    self.requestAPIs.removeValue(forKey: request.requestID)
                    if let hostID = self.requestHostIDs.removeValue(forKey: request.requestID) {
                        self.reservedRequests[hostID] = max(
                            0,
                            (self.reservedRequests[hostID] ?? 1) - 1
                        )
                    }
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    func cancelInference(requestID: String) async -> Bool {
        let target = requestAPIs[requestID] ?? inferenceAPI
        return (try? await target.cancel(requestID: requestID)) ?? false
    }

    func selectTeamHost(_ profile: TeamHostProfile) {
        guard let remote = teamAPIs[profile.id] else { return }
        teamHost = profile
        inferenceAPI = remote
        Self.saveTeamProfile(profile)
    }

    func removeTeamHost(_ profile: TeamHostProfile) {
        if teamHost?.id == profile.id {
            forgetTeamHost()
            return
        }
        teamHosts.removeAll { $0.id == profile.id }
        teamAPIs.removeValue(forKey: profile.id)
        teamHostSnapshots.removeValue(forKey: profile.id)
        Self.saveTeamProfiles(teamHosts)
        Task { try? await KeychainStore.deleteTeamTokenAsync(profileID: profile.id) }
        rebuildTeamCatalog()
    }

    private func rebuildAPI() {
#if DEBUG
        guard !usesUITestAPI else { return }
#endif
        api = MachBoostAPI(endpoint: configuration.endpoint, apiToken: apiToken)
        if inferenceMode == .local {
            inferenceAPI = api
        }
    }

    private func reconnectTeamHosts() async {
        guard !teamHosts.isEmpty else {
            useLocalInference()
            return
        }
        for profile in teamHosts {
            await refreshTeamHost(profile)
        }
        rebuildTeamCatalog()
        if let selected = teamHost, let remote = teamAPIs[selected.id] {
            inferenceAPI = remote
        } else if let profile = teamHosts.first(where: {
            teamHostSnapshots[$0.id]?.isOnline == true
        }), let remote = teamAPIs[profile.id] {
            teamHost = profile
            inferenceAPI = remote
            Self.saveTeamProfile(profile)
        } else if !includeLocalInHostPool {
            presentedError = "No configured MachBoost host is currently reachable."
        }
        startHeartbeat()
    }

    private func startHeartbeat() {
        heartbeatTask?.cancel()
        heartbeatTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                guard let self, !Task.isCancelled, self.inferenceMode == .team else {
                    return
                }
                await self.reconnectTeamHostsWithoutRestartingHeartbeat()
            }
        }
    }

    private func reconnectTeamHostsWithoutRestartingHeartbeat() async {
        for profile in teamHosts {
            await refreshTeamHost(profile)
        }
        rebuildTeamCatalog()
    }

    private func refreshTeamHost(_ profile: TeamHostProfile) async {
        guard let token = await KeychainStore.teamTokenAsync(profileID: profile.id) else {
            teamHostSnapshots[profile.id] = TeamHostSnapshot(
                profile: profile,
                catalog: [],
                loadedModels: [],
                metrics: nil,
                isOnline: false,
                lastError: "Missing API key",
                updatedAt: .now
            )
            return
        }
        let remote = MachBoostAPI(
            endpoint: profile.endpoint,
            apiToken: token,
            deviceID: deviceID
        )
        do {
            async let connected = remote.teamConnect()
            async let metrics = remote.metrics()
            let values = try await (connected, metrics)
            let updated = TeamHostProfile(
                id: profile.id,
                endpoint: profile.endpoint,
                hostName: values.0.host.name,
                hostVersion: values.0.host.version,
                principalName: values.0.principal.name,
                connectedAt: profile.connectedAt
            )
            if let index = teamHosts.firstIndex(where: { $0.id == profile.id }) {
                teamHosts[index] = updated
            }
            teamAPIs[profile.id] = remote
            teamHostSnapshots[profile.id] = TeamHostSnapshot(
                profile: updated,
                catalog: values.0.models,
                loadedModels: values.0.loadedModels,
                metrics: values.1,
                isOnline: true,
                lastError: nil,
                updatedAt: .now
            )
            if Date().timeIntervalSince(lastPresenceAt[profile.id] ?? .distantPast) >= 30 {
                _ = try? await remote.reportTeamPresence(
                    deviceID: deviceID,
                    deviceName: Host.current().localizedName ?? ProcessInfo.processInfo.hostName,
                    appVersion: Bundle.main.object(
                        forInfoDictionaryKey: "CFBundleShortVersionString"
                    ) as? String ?? "development",
                    workspaceName: nil,
                    workspaceFingerprint: nil,
                    model: nil
                )
                lastPresenceAt[profile.id] = .now
            }
        } catch {
            teamHostSnapshots[profile.id] = TeamHostSnapshot(
                profile: profile,
                catalog: teamHostSnapshots[profile.id]?.catalog ?? [],
                loadedModels: teamHostSnapshots[profile.id]?.loadedModels ?? [],
                metrics: teamHostSnapshots[profile.id]?.metrics,
                isOnline: false,
                lastError: error.localizedDescription,
                updatedAt: .now
            )
            teamAPIs.removeValue(forKey: profile.id)
        }
    }

    private func rebuildTeamCatalog() {
        let online = teamHostSnapshots.values.filter(\.isOnline)
        var models: [String: CatalogModel] = [:]
        for model in online.flatMap(\.catalog) {
            models[model.name] = model
        }
        teamCatalog = models.values.sorted {
            $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending
        }
        var loaded: [String: ModelInstance] = [:]
        for model in online.flatMap(\.loadedModels) {
            loaded[model.id] = model
        }
        teamLoadedModels = Array(loaded.values)
        Self.saveTeamProfiles(teamHosts)
    }

    private func inferenceSelection(
        for model: String
    ) throws -> (api: any MachBoostAPIProtocol, hostID: UUID) {
        guard inferenceMode == .team else { return (api, Self.localPoolID) }
        var candidates: [(
            score: Double,
            api: any MachBoostAPIProtocol,
            profile: TeamHostProfile?,
            hostID: UUID
        )] = []
        if includeLocalInHostPool,
           catalog.contains(where: {
               ($0.name == model || $0.repository == model) && $0.cached && $0.support == "ready"
           }) {
            let loaded = loadedModels.contains { $0.model == model }
            let score = HostRoutingPolicy.score(
                metrics: metrics,
                modelLoaded: loaded,
                reservedRequests: reservedRequests[Self.localPoolID] ?? 0
            )
            candidates.append((score, api, nil, Self.localPoolID))
        }
        for snapshot in teamHostSnapshots.values where snapshot.isOnline && snapshot.supports(model: model) {
            guard let remote = teamAPIs[snapshot.id] else { continue }
            candidates.append((
                HostRoutingPolicy.score(
                    metrics: snapshot.metrics,
                    modelLoaded: snapshot.hasLoaded(model: model),
                    reservedRequests: reservedRequests[snapshot.id] ?? 0
                ),
                remote,
                snapshot.profile,
                snapshot.id
            ))
        }
        guard let selected = candidates.min(by: { $0.score < $1.score }) else {
            throw AppStateError.invalidTeamHost(
                "No online host has \(model) ready. Download it on a host or enable this Mac in the pool."
            )
        }
        if let profile = selected.profile {
            teamHost = profile
            inferenceAPI = selected.api
            Self.saveTeamProfile(profile)
        }
        return (selected.api, selected.hostID)
    }

    private func updateHostAdvertisement() {
        if configuration.lanEnabled {
            hostDiscovery.publish(
                name: Host.current().localizedName ?? ProcessInfo.processInfo.hostName,
                port: configuration.port
            )
        } else {
            hostDiscovery.stopPublishing()
        }
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

    private static func loadTeamProfile() -> TeamHostProfile? {
        guard let data = UserDefaults.standard.data(forKey: teamProfileKey) else {
            return nil
        }
        return try? JSONDecoder().decode(TeamHostProfile.self, from: data)
    }

    private static func saveTeamProfile(_ profile: TeamHostProfile) {
        guard let data = try? JSONEncoder().encode(profile) else { return }
        UserDefaults.standard.set(data, forKey: teamProfileKey)
    }

    private static func loadTeamProfiles(
        fallback: TeamHostProfile?
    ) -> [TeamHostProfile] {
        guard
            let data = UserDefaults.standard.data(forKey: teamProfilesKey),
            let profiles = try? JSONDecoder().decode([TeamHostProfile].self, from: data)
        else {
            return fallback.map { [$0] } ?? []
        }
        return profiles
    }

    private static func saveTeamProfiles(_ profiles: [TeamHostProfile]) {
        guard let data = try? JSONEncoder().encode(profiles) else { return }
        UserDefaults.standard.set(data, forKey: teamProfilesKey)
    }

    private static func saveInferenceMode(_ mode: InferenceMode) {
        UserDefaults.standard.set(mode.rawValue, forKey: inferenceModeKey)
    }

    private static func loadDeviceID() -> String {
        if let existing = UserDefaults.standard.string(forKey: deviceIDKey),
           !existing.isEmpty {
            return existing
        }
        let created = UUID().uuidString.lowercased()
        UserDefaults.standard.set(created, forKey: deviceIDKey)
        return created
    }

    private static func normalizedTeamEndpoint(_ value: String) throws -> URL {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let candidate = trimmed.contains("://") ? trimmed : "http://\(trimmed)"
        guard var components = URLComponents(string: candidate),
              ["http", "https"].contains(components.scheme?.lowercased() ?? ""),
              components.host != nil else {
            throw AppStateError.invalidTeamHost("Enter a valid HTTP or HTTPS host URL.")
        }
        components.path = ""
        components.query = nil
        components.fragment = nil
        guard let url = components.url else {
            throw AppStateError.invalidTeamHost("Enter a valid Team host URL.")
        }
        return url
    }
}

enum AppStateError: LocalizedError {
    case serverNotRunning
    case unsupportedModel(String)
    case invalidTeamHost(String)

    var errorDescription: String? {
        switch self {
        case .serverNotRunning:
            "Start the MachBoost server before loading a model."
        case let .unsupportedModel(reason):
            "This model is not compatible with the bundled MLX runtime: \(reason)"
        case let .invalidTeamHost(reason):
            reason
        }
    }
}
