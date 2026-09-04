import Foundation
import MachBoostDaemonClient
import Observation

private struct InferenceCandidate: Sendable {
    let score: Double
    let api: any MachBoostAPIProtocol
    let profile: TeamHostProfile?
    let hostID: UUID
    let hostName: String
}

struct InferenceRouteRecord: Sendable {
    let hostID: String
    let hostName: String
    let expectedDelay: Double
}

private struct TeamHostRefreshResult: Sendable {
    let profile: TeamHostProfile
    let api: MachBoostAPI?
    let connected: TeamConnectResponse?
    let metrics: ServerMetrics?
    let roundTripSeconds: Double
    let error: String?
}

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
    private(set) var mcpServers: [MCPServerSummary] = []
    private(set) var skills: [SkillSummary] = []
    private(set) var lastCreatedTeamToken: String?
    private(set) var downloads: [String: PullEvent] = [:]
    private(set) var loadingModels: Set<String> = []
    private(set) var deletingModels: Set<String> = []
    private(set) var lastModelLoad: ModelLoadResponse?
    private(set) var indexingWorkspaces: Set<String> = []
    private(set) var isRefreshing = false
    private var heartbeatTask: Task<Void, Never>?
    private var teamAPIs: [UUID: any MachBoostAPIProtocol] = [:]
    private var requestAPIs: [String: any MachBoostAPIProtocol] = [:]
    private var requestHostIDs: [String: UUID] = [:]
    private var completedRequestRoutes: [String: InferenceRouteRecord] = [:]
    private var reservedRequests: [UUID: Int] = [:]
    private var hostFailureCounts: [UUID: Int] = [:]
    private var hostCooldownUntil: [UUID: Date] = [:]
    private var lastPresenceAt: [UUID: Date] = [:]
    private(set) var lastRoutedHostID: UUID?
    private(set) var lastRouteExpectedDelay: Double?
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
        let loadedProfiles = isTesting ? [] : Self.loadTeamProfiles(fallback: storedProfile)
        let storedProfiles = loadedProfiles.filter {
            !Self.isLocalTeamEndpoint($0.endpoint)
        }
        let selectedProfile = storedProfile.flatMap { selected in
            storedProfiles.first { $0.id == selected.id }
        } ?? storedProfiles.first
        if !isTesting, storedProfiles.count != loadedProfiles.count {
            Self.saveTeamProfiles(storedProfiles)
            if let selectedProfile {
                Self.saveTeamProfile(selectedProfile)
            } else {
                UserDefaults.standard.removeObject(forKey: Self.teamProfileKey)
            }
        }
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
        inferenceMode == .team && hasOnlineTeamHost ? teamCatalog : catalog
    }

    var activeLoadedModels: [ModelInstance] {
        inferenceMode == .team && hasOnlineTeamHost ? teamLoadedModels : loadedModels
    }

    var inferenceLabel: String {
        inferencePresentation.destination
    }

    var inferenceStatusLabel: String {
        inferencePresentation.status
    }

    var teamIsConnected: Bool {
        inferenceMode == .team && hasOnlineTeamHost
    }

    var includeLocalInHostPool: Bool {
        get { UserDefaults.standard.object(forKey: Self.includeLocalPoolKey) as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: Self.includeLocalPoolKey) }
    }

    var lastRouteWasLocal: Bool {
        lastRoutedHostID == Self.localPoolID
    }

    var lastRoutedHostName: String? {
        guard let lastRoutedHostID else { return nil }
        if lastRoutedHostID == Self.localPoolID {
            return Host.current().localizedName ?? "This Mac"
        }
        return teamHosts.first { $0.id == lastRoutedHostID }?.hostName
    }

    func wasLastRouted(to profile: TeamHostProfile) -> Bool {
        lastRoutedHostID == profile.id
    }

    func connectToTeamHost(endpoint rawEndpoint: String, token: String) async {
        do {
            let endpoint = try Self.normalizedTeamEndpoint(rawEndpoint)
            guard !Self.isLocalTeamEndpoint(endpoint) else {
                throw AppStateError.invalidTeamHost(
                    "This address belongs to this Mac. Use This Mac in the inference pool instead."
                )
            }
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
            let started = Date()
            async let connectedRequest = remote.teamConnect()
            async let metricsRequest = remote.metrics()
            let (connected, hostMetrics) = try await (connectedRequest, metricsRequest)
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
                metrics: hostMetrics,
                roundTripSeconds: Date().timeIntervalSince(started),
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
        let deviceName = Host.current().localizedName ?? ProcessInfo.processInfo.hostName
        let appVersion = Bundle.main.object(
            forInfoDictionaryKey: "CFBundleShortVersionString"
        ) as? String ?? "development"
        var firstResponse: TeamClient?
        var lastError: Error?
        for profile in teamHosts {
            guard let remote = teamAPIs[profile.id] else { continue }
            do {
                let response = try await remote.reportTeamPresence(
                    deviceID: deviceID,
                    deviceName: deviceName,
                    appVersion: appVersion,
                    workspaceName: workspace?.name,
                    workspaceFingerprint: workspace?.revision,
                    model: model
                )
                firstResponse = firstResponse ?? response
                lastPresenceAt[profile.id] = .now
            } catch {
                lastError = error
            }
        }
        if let firstResponse {
            teamClient = firstResponse
        } else if let lastError {
            throw lastError
        }
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
            if daemon.authenticationRequired, !configuration.lanEnabled {
                configuration.lanEnabled = true
                Self.saveConfiguration(configuration)
            }
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
            teamClients = Self.deduplicatedTeamClients(
                values.4,
                localDeviceID: deviceID,
                localDeviceName: Host.current().localizedName
                    ?? ProcessInfo.processInfo.hostName
            )
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
            async let extensions = api.extensions()
            let values = try await (memories, metrics, providers, extensions)
            self.memories = values.0
            cacheMetrics = values.1
            self.providers = values.2
            mcpServers = values.3.mcpServers
            skills = values.3.skills
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
            mcpServers = []
            skills = []
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func configureMCPServer(
        id: String?,
        name: String,
        transport: String,
        url: String?,
        command: String?,
        args: [String],
        environment: [String: String],
        headers: [String: String],
        enabled: Bool
    ) async -> Bool {
        do {
            _ = try await api.configureMCPServer(
                id: id,
                name: name,
                transport: transport,
                url: url,
                command: command,
                args: args,
                environment: environment,
                headers: headers,
                enabled: enabled
            )
            await refreshMemoryAndProviders()
            return true
        } catch {
            presentedError = error.localizedDescription
            return false
        }
    }

    func deleteMCPServer(id: String) async {
        do {
            try await api.deleteMCPServer(id: id)
            await refreshMemoryAndProviders()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func testMCPServer(id: String) async -> [MCPToolSummary]? {
        do {
            let tools = try await api.testMCPServer(id: id)
            await refreshMemoryAndProviders()
            return tools
        } catch {
            presentedError = error.localizedDescription
            await refreshMemoryAndProviders()
            return nil
        }
    }

    func configureSkill(
        id: String?,
        name: String,
        instructions: String,
        enabled: Bool
    ) async -> Bool {
        do {
            _ = try await api.configureSkill(
                id: id,
                name: name,
                instructions: instructions,
                enabled: enabled
            )
            await refreshMemoryAndProviders()
            return true
        } catch {
            presentedError = error.localizedDescription
            return false
        }
    }

    func deleteSkill(id: String) async {
        do {
            try await api.deleteSkill(id: id)
            await refreshMemoryAndProviders()
        } catch {
            presentedError = error.localizedDescription
        }
    }

    func searchMCPTools(query: String, limit: Int = 8) async throws -> [MCPToolSummary] {
        try await api.searchMCPTools(query: query, limit: limit)
    }

    func callMCPTool(
        serverID: String,
        name: String,
        arguments: JSONValue
    ) async throws -> MCPToolResult {
        try await api.callMCPTool(
            serverID: serverID,
            name: name,
            arguments: arguments
        )
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
        guard downloads[model] == nil else { return }
        let requestID = "pull-\(UUID().uuidString.lowercased())"
        downloads[model] = PullEvent(
            requestID: requestID,
            status: "Checking compatibility",
            file: nil,
            completed: nil,
            total: nil,
            unit: nil,
            done: false,
            path: nil,
            error: nil
        )
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

    func deleteModel(_ model: String) async {
        guard serverIsRunning else {
            presentedError = AppStateError.serverNotRunning.localizedDescription
            return
        }
        guard downloads[model] == nil else {
            presentedError = "Cancel the active model download before deleting it."
            return
        }
        deletingModels.insert(model)
        defer { deletingModels.remove(model) }
        do {
            let response = try await api.deleteModel(model: model)
            guard response.removed else {
                throw AppStateError.modelNotFound(model)
            }
            await refreshAll()
        } catch {
            presentedError = error.localizedDescription
        }
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

    func searchHubModels(query: String, limit: Int = 16) async -> [CatalogModel] {
        let normalized = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard normalized.count >= 2, serverIsRunning else { return [] }
        do {
            return try await api.searchCatalog(query: normalized, limit: limit)
        } catch {
            return []
        }
    }

    func workspace(id: String?) -> WorkspaceSummary? {
        guard let id else { return nil }
        return workspaces.first { $0.id == id }
    }

    func streamChat(
        _ request: ChatRequest,
        preferredHostID: String? = nil
    ) throws -> AsyncThrowingStream<ChatEvent, Error> {
        let candidates = try inferenceCandidates(
            for: request.model,
            preferredHostID: preferredHostID
        )
        return AsyncThrowingStream { continuation in
            let task = Task { @MainActor [weak self] in
                guard let self else {
                    continuation.finish(throwing: CancellationError())
                    return
                }
                var lastError: Error?
                for (index, selected) in candidates.enumerated() {
                    if Task.isCancelled {
                        continuation.finish()
                        return
                    }
                    self.beginRequestRoute(requestID: request.requestID, candidate: selected)
                    var emittedOutput = false
                    var receivedDone = false
                    do {
                        for try await event in selected.api.streamChat(request) {
                            if !emittedOutput, let message = event.error, !message.isEmpty {
                                throw MachBoostAPIError.stream(message)
                            }
                            let visible = Self.hasVisibleOutput(event)
                            emittedOutput = emittedOutput || visible
                            receivedDone = receivedDone || event.done
                            if visible || event.done {
                                continuation.yield(event)
                            }
                        }
                        if !emittedOutput && !receivedDone {
                            throw MachBoostAPIError.stream(
                                "Host closed the response before producing output."
                            )
                        }
                        self.markRouteSuccessful(selected)
                        self.recordCompletedRoute(
                            requestID: request.requestID,
                            candidate: selected
                        )
                        self.endRequestRoute(
                            requestID: request.requestID,
                            hostID: selected.hostID
                        )
                        continuation.finish()
                        return
                    } catch {
                        lastError = error
                        _ = try? await selected.api.cancel(requestID: request.requestID)
                        self.markRouteFailed(selected)
                        self.endRequestRoute(
                            requestID: request.requestID,
                            hostID: selected.hostID
                        )
                        let hasAnotherHost = index + 1 < candidates.count
                        guard hasAnotherHost,
                              HostRoutingPolicy.canFailOver(
                                  error: error,
                                  emittedOutput: emittedOutput
                              ) else {
                            if error is CancellationError || Task.isCancelled {
                                continuation.finish()
                            } else {
                                continuation.finish(throwing: error)
                            }
                            return
                        }
                    }
                }
                continuation.finish(throwing: lastError ?? MachBoostAPIError.invalidResponse)
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    func cancelInference(requestID: String) async -> Bool {
        let target = requestAPIs[requestID] ?? inferenceAPI
        return (try? await target.cancel(requestID: requestID)) ?? false
    }

    func consumeInferenceRoute(requestID: String) -> InferenceRouteRecord? {
        completedRequestRoutes.removeValue(forKey: requestID)
    }

    func inferenceHostOptions(for model: String) -> [InferenceHostOption] {
        var result = [
            InferenceHostOption(
                id: InferenceHostOption.automaticID,
                name: "Automatic",
                detail: "Use the fastest ready device",
                isOnline: true,
                isLoaded: false
            )
        ]
        let localReady = catalog.contains {
            ($0.name == model || $0.repository == model) && $0.cached && $0.support == "ready"
        }
        result.append(
            InferenceHostOption(
                id: InferenceHostOption.localID,
                name: Host.current().localizedName ?? "This Mac",
                detail: loadedModels.contains(where: { $0.model == model })
                    ? "Loaded on this Mac"
                    : (localReady ? "Ready on this Mac" : "Model unavailable"),
                isOnline: true,
                isLoaded: loadedModels.contains { $0.model == model }
            )
        )
        result.append(contentsOf: teamHosts.map { profile in
            let snapshot = teamHostSnapshots[profile.id]
            let latency = snapshot.map {
                " · \(Int(($0.roundTripSeconds * 1_000).rounded())) ms"
            } ?? ""
            return InferenceHostOption(
                id: profile.id.uuidString,
                name: profile.hostName,
                detail: snapshot?.hasLoaded(model: model) == true
                    ? "Loaded\(latency)"
                    : (snapshot?.supports(model: model) == true
                        ? "Ready\(latency)"
                        : (snapshot?.isOnline == true ? "Model unavailable" : "Offline")),
                isOnline: snapshot?.isOnline == true,
                isLoaded: snapshot?.hasLoaded(model: model) == true
            )
        })
        return result
    }

    func selectTeamHost(_ profile: TeamHostProfile) {
        guard let remote = teamAPIs[profile.id] else { return }
        teamHost = profile
        inferenceMode = .team
        inferenceAPI = remote
        Self.saveTeamProfile(profile)
        Self.saveInferenceMode(.team)
        startHeartbeat()
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
        await refreshTeamHostsConcurrently()
        rebuildTeamCatalog()
        selectReachableInferenceAPI()
        startHeartbeat()
    }

    private func startHeartbeat() {
        heartbeatTask?.cancel()
        heartbeatTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                guard let self, !Task.isCancelled, self.inferenceMode == .team else {
                    return
                }
                await self.reconnectTeamHostsWithoutRestartingHeartbeat()
            }
        }
    }

    private func reconnectTeamHostsWithoutRestartingHeartbeat() async {
        await refreshTeamHostsConcurrently()
        rebuildTeamCatalog()
        selectReachableInferenceAPI()
    }

    private var hasOnlineTeamHost: Bool {
        teamHostSnapshots.values.contains(where: \.isOnline)
    }

    private var inferencePresentation: (destination: String, status: String) {
        let onlineProfiles = teamHosts.filter {
            teamHostSnapshots[$0.id]?.isOnline == true
        }
        let selectedOnlineName = teamHost.flatMap { selected in
            onlineProfiles.first(where: { $0.id == selected.id })?.hostName
        }
        return Self.inferencePresentation(
            mode: inferenceMode,
            serverIsRunning: serverIsRunning,
            onlineHostNames: onlineProfiles.map(\.hostName),
            selectedOnlineName: selectedOnlineName
        )
    }

    static func inferencePresentation(
        mode: InferenceMode,
        serverIsRunning: Bool,
        onlineHostNames: [String],
        selectedOnlineName: String?
    ) -> (destination: String, status: String) {
        guard mode == .team else {
            return (
                "This Mac",
                serverIsRunning ? "Local ready" : "Local offline"
            )
        }
        guard !onlineHostNames.isEmpty else {
            return (
                "This Mac",
                serverIsRunning
                    ? "Local fallback \u{00b7} remote unavailable"
                    : "Remote unavailable"
            )
        }
        let destination: String
        if onlineHostNames.count > 1 {
            destination = "Host pool (\(onlineHostNames.count))"
        } else {
            destination = selectedOnlineName ?? onlineHostNames[0]
        }
        return (destination, destination)
    }

    private func selectReachableInferenceAPI() {
        guard inferenceMode == .team else {
            inferenceAPI = api
            return
        }
        if let selected = teamHost,
           teamHostSnapshots[selected.id]?.isOnline == true,
           let remote = teamAPIs[selected.id] {
            inferenceAPI = remote
            return
        }
        if let profile = teamHosts.first(where: {
            teamHostSnapshots[$0.id]?.isOnline == true
        }), let remote = teamAPIs[profile.id] {
            teamHost = profile
            inferenceAPI = remote
            Self.saveTeamProfile(profile)
            return
        }
        inferenceAPI = api
    }

    private func refreshTeamHostsConcurrently() async {
        let profiles = teamHosts
        let currentDeviceID = deviceID
        let deviceName = Host.current().localizedName ?? ProcessInfo.processInfo.hostName
        let appVersion = Bundle.main.object(
            forInfoDictionaryKey: "CFBundleShortVersionString"
        ) as? String ?? "development"
        await withTaskGroup(of: TeamHostRefreshResult.self) { group in
            for profile in profiles {
                let token = await KeychainStore.teamTokenAsync(profileID: profile.id)
                let shouldReport = Date().timeIntervalSince(
                    lastPresenceAt[profile.id] ?? .distantPast
                ) >= 30
                if shouldReport {
                    lastPresenceAt[profile.id] = .now
                }
                group.addTask {
                    await Self.fetchTeamHost(
                        profile,
                        token: token,
                        deviceID: currentDeviceID,
                        deviceName: deviceName,
                        appVersion: appVersion,
                        reportPresence: shouldReport
                    )
                }
            }
            for await result in group {
                applyTeamHostRefresh(result)
            }
        }
    }

    private nonisolated static func fetchTeamHost(
        _ profile: TeamHostProfile,
        token: String?,
        deviceID: String,
        deviceName: String,
        appVersion: String,
        reportPresence: Bool
    ) async -> TeamHostRefreshResult {
        guard let token else {
            return TeamHostRefreshResult(
                profile: profile,
                api: nil,
                connected: nil,
                metrics: nil,
                roundTripSeconds: 0,
                error: "Missing API key"
            )
        }
        let remote = MachBoostAPI(
            endpoint: profile.endpoint,
            apiToken: token,
            deviceID: deviceID
        )
        let started = Date()
        do {
            async let connected = remote.teamConnect()
            async let metrics = remote.metrics()
            let values = try await (connected, metrics)
            if reportPresence {
                _ = try? await remote.reportTeamPresence(
                    deviceID: deviceID,
                    deviceName: deviceName,
                    appVersion: appVersion,
                    workspaceName: nil,
                    workspaceFingerprint: nil,
                    model: nil
                )
            }
            return TeamHostRefreshResult(
                profile: profile,
                api: remote,
                connected: values.0,
                metrics: values.1,
                roundTripSeconds: Date().timeIntervalSince(started),
                error: nil
            )
        } catch {
            return TeamHostRefreshResult(
                profile: profile,
                api: nil,
                connected: nil,
                metrics: nil,
                roundTripSeconds: Date().timeIntervalSince(started),
                error: error.localizedDescription
            )
        }
    }

    private func applyTeamHostRefresh(_ result: TeamHostRefreshResult) {
        let profile = result.profile
        guard teamHosts.contains(where: { $0.id == profile.id }) else { return }
        if let connected = result.connected, let remote = result.api {
            let updated = TeamHostProfile(
                id: profile.id,
                endpoint: profile.endpoint,
                hostName: connected.host.name,
                hostVersion: connected.host.version,
                principalName: connected.principal.name,
                connectedAt: profile.connectedAt
            )
            if let index = teamHosts.firstIndex(where: { $0.id == profile.id }) {
                teamHosts[index] = updated
            }
            teamAPIs[profile.id] = remote
            teamHostSnapshots[profile.id] = TeamHostSnapshot(
                profile: updated,
                catalog: connected.models,
                loadedModels: connected.loadedModels,
                metrics: result.metrics,
                roundTripSeconds: smoothedRoundTrip(
                    previous: teamHostSnapshots[profile.id]?.roundTripSeconds,
                    observed: result.roundTripSeconds
                ),
                isOnline: true,
                lastError: nil,
                updatedAt: .now
            )
        } else {
            teamHostSnapshots[profile.id] = TeamHostSnapshot(
                profile: profile,
                catalog: teamHostSnapshots[profile.id]?.catalog ?? [],
                loadedModels: teamHostSnapshots[profile.id]?.loadedModels ?? [],
                metrics: teamHostSnapshots[profile.id]?.metrics,
                roundTripSeconds: teamHostSnapshots[profile.id]?.roundTripSeconds ?? 0,
                isOnline: false,
                lastError: result.error,
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

    private func inferenceCandidates(
        for model: String,
        preferredHostID: String? = nil
    ) throws -> [InferenceCandidate] {
        guard inferenceMode == .team else {
            return [
                InferenceCandidate(
                    score: 0,
                    api: api,
                    profile: nil,
                    hostID: Self.localPoolID,
                    hostName: Host.current().localizedName ?? "This Mac"
                )
            ]
        }
        let now = Date()
        let prefersLocal = preferredHostID == InferenceHostOption.localID
        let preferredRemoteID = preferredHostID.flatMap { UUID(uuidString: $0) }
        var candidates: [InferenceCandidate] = []
        let hasEligibleRemote = teamHostSnapshots.values.contains { snapshot in
            snapshot.isOnline
                && snapshot.supports(model: model)
                && hostCooldownUntil[snapshot.id, default: .distantPast] <= now
        }
        if HostRoutingPolicy.shouldIncludeLocal(
            includeLocalInPool: includeLocalInHostPool,
            prefersLocal: prefersLocal,
            hasOnlineRemote: hasEligibleRemote
        ),
           hostCooldownUntil[Self.localPoolID, default: .distantPast] <= now,
           catalog.contains(where: {
               ($0.name == model || $0.repository == model) && $0.cached && $0.support == "ready"
           }) {
            let instance = loadedModels.first { $0.model == model }
            let score = HostRoutingPolicy.score(
                metrics: metrics,
                modelLoaded: instance != nil,
                reservedRequests: reservedRequests[Self.localPoolID] ?? 0,
                replicas: instance?.scheduler.replicas ?? 1,
                activeRequests: instance?.scheduler.activeRequests,
                queuedRequests: instance?.scheduler.queuedRequests
            )
            candidates.append(
                InferenceCandidate(
                    score: score,
                    api: api,
                    profile: nil,
                    hostID: Self.localPoolID,
                    hostName: Host.current().localizedName ?? "This Mac"
                )
            )
        }
        for snapshot in teamHostSnapshots.values
        where snapshot.isOnline
            && snapshot.supports(model: model)
            && hostCooldownUntil[snapshot.id, default: .distantPast] <= now {
            guard let remote = teamAPIs[snapshot.id] else { continue }
            let scheduler = snapshot.scheduler(model: model)
            candidates.append(
                InferenceCandidate(
                    score: HostRoutingPolicy.score(
                    metrics: snapshot.metrics,
                    modelLoaded: snapshot.hasLoaded(model: model),
                    reservedRequests: reservedRequests[snapshot.id] ?? 0,
                    roundTripSeconds: snapshot.roundTripSeconds,
                    replicas: scheduler?.replicas ?? 1,
                    activeRequests: scheduler?.activeRequests,
                    queuedRequests: scheduler?.queuedRequests
                    ),
                    api: remote,
                    profile: snapshot.profile,
                    hostID: snapshot.id,
                    hostName: snapshot.profile.hostName
                )
            )
        }
        guard !candidates.isEmpty else {
            throw AppStateError.invalidTeamHost(
                "No online host has \(model) ready. Download it on a host or enable this Mac in the pool."
            )
        }
        return candidates.sorted {
            let lhsPreferred = prefersLocal
                ? $0.hostID == Self.localPoolID
                : $0.hostID == preferredRemoteID
            let rhsPreferred = prefersLocal
                ? $1.hostID == Self.localPoolID
                : $1.hostID == preferredRemoteID
            if lhsPreferred != rhsPreferred {
                return lhsPreferred
            }
            if $0.score == $1.score {
                return $0.hostID.uuidString < $1.hostID.uuidString
            }
            return $0.score < $1.score
        }
    }

    private func beginRequestRoute(requestID: String, candidate: InferenceCandidate) {
        requestAPIs[requestID] = candidate.api
        requestHostIDs[requestID] = candidate.hostID
        reservedRequests[candidate.hostID, default: 0] += 1
    }

    private func endRequestRoute(requestID: String, hostID: UUID) {
        if requestHostIDs[requestID] == hostID {
            requestAPIs.removeValue(forKey: requestID)
            requestHostIDs.removeValue(forKey: requestID)
        }
        reservedRequests[hostID] = max(0, (reservedRequests[hostID] ?? 1) - 1)
    }

    private func markRouteSuccessful(_ candidate: InferenceCandidate) {
        hostFailureCounts[candidate.hostID] = 0
        hostCooldownUntil[candidate.hostID] = nil
        lastRoutedHostID = candidate.hostID
        lastRouteExpectedDelay = candidate.score
        inferenceAPI = candidate.api
        if let profile = candidate.profile {
            teamHost = profile
            Self.saveTeamProfile(profile)
        }
    }

    private func recordCompletedRoute(requestID: String, candidate: InferenceCandidate) {
        completedRequestRoutes[requestID] = InferenceRouteRecord(
            hostID: candidate.hostID == Self.localPoolID
                ? InferenceHostOption.localID
                : candidate.hostID.uuidString,
            hostName: candidate.hostName,
            expectedDelay: candidate.score
        )
        if completedRequestRoutes.count > 128,
           let oldest = completedRequestRoutes.keys.sorted().first {
            completedRequestRoutes.removeValue(forKey: oldest)
        }
    }

    private func markRouteFailed(_ candidate: InferenceCandidate) {
        let failures = (hostFailureCounts[candidate.hostID] ?? 0) + 1
        hostFailureCounts[candidate.hostID] = failures
        let cooldown = min(60, 10 * pow(2, Double(max(0, failures - 1))))
        hostCooldownUntil[candidate.hostID] = Date().addingTimeInterval(cooldown)
    }

    nonisolated static func hasVisibleOutput(_ event: ChatEvent) -> Bool {
        let message = event.message
        return !(message?.content.isEmpty ?? true)
            || !(message?.thinking ?? "").isEmpty
            || !(message?.toolCalls ?? []).isEmpty
            || event.machboost?.fullContent != nil
    }

    private func smoothedRoundTrip(previous: Double?, observed: Double) -> Double {
        guard let previous, previous > 0 else { return max(0, observed) }
        return previous * 0.7 + max(0, observed) * 0.3
    }

    private func updateHostAdvertisement() {
        if configuration.lanEnabled {
            hostDiscovery.publish(
                name: Host.current().localizedName ?? ProcessInfo.processInfo.hostName,
                port: configuration.port,
                deviceID: deviceID
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

    static func deduplicatedTeamClients(
        _ clients: [TeamClient],
        localDeviceID: String,
        localDeviceName: String
    ) -> [TeamClient] {
        let localName = normalizedDeviceName(localDeviceName)
        var selected: [String: TeamClient] = [:]
        for client in clients {
            let name = normalizedDeviceName(client.deviceName)
            guard client.deviceID != localDeviceID, name != localName else { continue }
            let key = name.isEmpty ? client.deviceID.lowercased() : name
            guard let existing = selected[key] else {
                selected[key] = client
                continue
            }
            if shouldPreferTeamClient(client, over: existing) {
                selected[key] = client
            }
        }
        return Array(selected.values)
    }

    private static func normalizedDeviceName(_ value: String) -> String {
        value.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
            .unicodeScalars
            .filter(CharacterSet.alphanumerics.contains)
            .map(String.init)
            .joined()
    }

    private static func shouldPreferTeamClient(
        _ candidate: TeamClient,
        over existing: TeamClient
    ) -> Bool {
        if candidate.online != existing.online { return candidate.online }
        if candidate.lastSeenAt != existing.lastSeenAt {
            return candidate.lastSeenAt > existing.lastSeenAt
        }
        return candidate.requestCount > existing.requestCount
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

    static func isLocalTeamEndpoint(
        _ endpoint: URL,
        localNames: Set<String>? = nil,
        localAddresses: Set<String>? = nil
    ) -> Bool {
        guard let rawHost = endpoint.host else { return false }
        let host = normalizedHost(rawHost)
        var names = localNames ?? [
            ProcessInfo.processInfo.hostName,
            Host.current().name ?? "",
        ]
        names.formUnion(["localhost", "localhost.localdomain"])
        let addresses = localAddresses ?? Set(Host.current().addresses)
        let normalizedNames = Set(names.map(normalizedHost).filter { !$0.isEmpty })
        let normalizedAddresses = Set(addresses.map(normalizedHost).filter { !$0.isEmpty })
        return normalizedNames.contains(host)
            || normalizedAddresses.contains(host)
            || host == "127.0.0.1"
            || host == "::1"
    }

    private static func normalizedHost(_ value: String) -> String {
        value.trimmingCharacters(in: CharacterSet(charactersIn: "[] "))
            .trimmingCharacters(in: CharacterSet(charactersIn: "."))
            .lowercased()
    }
}

enum AppStateError: LocalizedError {
    case serverNotRunning
    case unsupportedModel(String)
    case modelNotFound(String)
    case invalidTeamHost(String)

    var errorDescription: String? {
        switch self {
        case .serverNotRunning:
            "Start the MachBoost server before loading a model."
        case let .unsupportedModel(reason):
            "This model is not compatible with the bundled MLX runtime: \(reason)"
        case let .modelNotFound(model):
            "The downloaded files for \(model) were not found."
        case let .invalidTeamHost(reason):
            reason
        }
    }
}
