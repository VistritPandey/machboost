import Foundation

public struct ServerConfiguration: Codable, Equatable, Sendable {
    public var lanEnabled: Bool
    public var port: Int
    public var replicas: Int
    public var maxQueue: Int
    public var queueTimeout: Double
    public var launchAtLogin: Bool

    public init(
        lanEnabled: Bool = false,
        port: Int = 11_435,
        replicas: Int = 1,
        maxQueue: Int = 64,
        queueTimeout: Double = 300,
        launchAtLogin: Bool = false
    ) {
        self.lanEnabled = lanEnabled
        self.port = port
        self.replicas = replicas
        self.maxQueue = maxQueue
        self.queueTimeout = queueTimeout
        self.launchAtLogin = launchAtLogin
    }

    public var bindHost: String { lanEnabled ? "0.0.0.0" : "127.0.0.1" }
    public var clientHost: String { "127.0.0.1" }
    public var endpoint: URL { URL(string: "http://\(clientHost):\(port)")! }
    public var advertisedEndpoint: URL {
        guard lanEnabled else { return endpoint }
        var components = URLComponents()
        components.scheme = "http"
        components.host = ProcessInfo.processInfo.hostName
        components.port = port
        return components.url ?? endpoint
    }
}

public struct TeamSettings: Codable, Equatable, Sendable {
    public let traceMode: String
    public let retentionDays: Int?
    public let maxStorageBytes: Int64

    public init(traceMode: String, retentionDays: Int?, maxStorageBytes: Int64) {
        self.traceMode = traceMode
        self.retentionDays = retentionDays
        self.maxStorageBytes = maxStorageBytes
    }

    enum CodingKeys: String, CodingKey {
        case traceMode = "trace_mode"
        case retentionDays = "retention_days"
        case maxStorageBytes = "max_storage_bytes"
    }
}

public struct TeamStatus: Decodable, Sendable {
    public let schema: String
    public let keys: Int
    public let traces: Int
    public let evaluations: Int
    public let settings: TeamSettings
}

public struct TeamKey: Codable, Identifiable, Hashable, Sendable {
    public let id: String
    public let name: String
    public let kind: String
    public let scopes: [String]
    public let allowedModels: [String]
    public let maxConcurrent: Int
    public let requestsPerMinute: Int
    public let enabled: Bool?
    public let createdAt: String?
    public let lastUsedAt: String?

    enum CodingKeys: String, CodingKey {
        case id, name, kind, scopes, enabled
        case allowedModels = "allowed_models"
        case maxConcurrent = "max_concurrent"
        case requestsPerMinute = "requests_per_minute"
        case createdAt = "created_at"
        case lastUsedAt = "last_used_at"
    }
}

public struct TeamKeysResponse: Decodable, Sendable {
    public let schema: String
    public let keys: [TeamKey]
}

public struct CreatedTeamKeyResponse: Decodable, Sendable {
    public let schema: String
    public let token: String
    public let key: TeamKey
}

public struct TracePrincipal: Codable, Hashable, Sendable {
    public let id: String
    public let name: String
}

public struct TraceSummary: Codable, Identifiable, Hashable, Sendable {
    public let id: String
    public let requestID: String
    public let principal: TracePrincipal
    public let endpoint: String
    public let model: String
    public let status: String
    public let startedAt: String
    public let durationSeconds: Double
    public let promptTokens: Int
    public let completionTokens: Int
    public let timeToFirstTokenSeconds: Double?

    enum CodingKeys: String, CodingKey {
        case id, principal, endpoint, model, status
        case requestID = "request_id"
        case startedAt = "started_at"
        case durationSeconds = "duration_seconds"
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case timeToFirstTokenSeconds = "time_to_first_token_seconds"
    }
}

public struct TracesResponse: Decodable, Sendable {
    public let schema: String
    public let traces: [TraceSummary]
}

public struct EvaluationLatency: Codable, Sendable {
    public let p50: Double
    public let p95: Double
}

public struct EvaluationSummary: Codable, Sendable {
    public let traceCount: Int
    public let completionRate: Double
    public let latencySeconds: EvaluationLatency
    public let timeToFirstTokenSeconds: EvaluationLatency
    public let generationTokensPerSecond: Double

    enum CodingKeys: String, CodingKey {
        case traceCount = "trace_count"
        case completionRate = "completion_rate"
        case latencySeconds = "latency_seconds"
        case timeToFirstTokenSeconds = "time_to_first_token_seconds"
        case generationTokensPerSecond = "generation_tokens_per_second"
    }
}

public struct TraceEvaluation: Codable, Identifiable, Sendable {
    public let id: String
    public let name: String
    public let evaluator: String
    public let traceIDs: [String]
    public let summary: EvaluationSummary
    public let createdAt: String

    enum CodingKeys: String, CodingKey {
        case id, name, evaluator, summary
        case traceIDs = "trace_ids"
        case createdAt = "created_at"
    }
}

public struct EvaluationsResponse: Decodable, Sendable {
    public let schema: String
    public let evaluations: [TraceEvaluation]
}

public struct EvaluationResponse: Decodable, Sendable {
    public let schema: String
    public let evaluation: TraceEvaluation
}

public struct CatalogResponse: Decodable, Sendable {
    public let schema: String
    public let models: [CatalogModel]
}

public struct CatalogModel: Codable, Identifiable, Hashable, Sendable {
    public let name: String
    public let displayName: String
    public let repository: String?
    public let backend: String
    public let capabilities: [String]
    public let cached: Bool
    public let cachedPath: String?
    public let recommended: Bool
    public let tested: Bool
    public let downloadSizeGB: Double?
    public let diskSizeGB: Double?
    public let minimumMemoryGB: Double?
    public let support: String
    public let supportReason: String?

    public init(
        name: String,
        displayName: String,
        repository: String?,
        backend: String,
        capabilities: [String],
        cached: Bool,
        cachedPath: String?,
        recommended: Bool,
        tested: Bool,
        downloadSizeGB: Double?,
        diskSizeGB: Double?,
        minimumMemoryGB: Double?,
        support: String,
        supportReason: String?
    ) {
        self.name = name
        self.displayName = displayName
        self.repository = repository
        self.backend = backend
        self.capabilities = capabilities
        self.cached = cached
        self.cachedPath = cachedPath
        self.recommended = recommended
        self.tested = tested
        self.downloadSizeGB = downloadSizeGB
        self.diskSizeGB = diskSizeGB
        self.minimumMemoryGB = minimumMemoryGB
        self.support = support
        self.supportReason = supportReason
    }

    public var id: String { name }
    public var supportsVision: Bool { capabilities.contains("vision") }

    enum CodingKeys: String, CodingKey {
        case name
        case displayName = "display_name"
        case repository
        case backend
        case capabilities
        case cached
        case cachedPath = "cached_path"
        case recommended
        case tested
        case downloadSizeGB = "download_size_gb"
        case diskSizeGB = "disk_size_gb"
        case minimumMemoryGB = "minimum_memory_gb"
        case support
        case supportReason = "support_reason"
    }
}

public struct APIChatMessage: Encodable, Sendable {
    public let role: String
    public let content: String
    public let images: [String]?

    public init(role: String, content: String, images: [String]?) {
        self.role = role
        self.content = content
        self.images = images
    }
}

public struct WorkspaceLanguage: Codable, Hashable, Sendable {
    public let name: String
    public let files: Int

    public init(name: String, files: Int) {
        self.name = name
        self.files = files
    }
}

public struct WorkspaceSummary: Codable, Identifiable, Hashable, Sendable {
    public let id: String
    public let name: String
    public let path: String
    public let createdAt: String
    public let updatedAt: String
    public let indexedAt: String?
    public let revision: String?
    public let fileCount: Int
    public let chunkCount: Int
    public let totalBytes: Int64
    public let languages: [WorkspaceLanguage]

    public init(
        id: String,
        name: String,
        path: String,
        createdAt: String,
        updatedAt: String,
        indexedAt: String?,
        revision: String?,
        fileCount: Int,
        chunkCount: Int,
        totalBytes: Int64,
        languages: [WorkspaceLanguage]
    ) {
        self.id = id
        self.name = name
        self.path = path
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.indexedAt = indexedAt
        self.revision = revision
        self.fileCount = fileCount
        self.chunkCount = chunkCount
        self.totalBytes = totalBytes
        self.languages = languages
    }

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case path
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case indexedAt = "indexed_at"
        case revision
        case fileCount = "file_count"
        case chunkCount = "chunk_count"
        case totalBytes = "total_bytes"
        case languages
    }
}

public struct WorkspacesResponse: Decodable, Sendable {
    public let schema: String
    public let workspaces: [WorkspaceSummary]
}

public struct WorkspaceIndexResponse: Decodable, Sendable {
    public let status: String
    public let workspace: WorkspaceSummary
    public let scannedFiles: Int?
    public let indexedFiles: Int?
    public let unchangedFiles: Int?
    public let removedFiles: Int?
    public let skippedFiles: Int?

    enum CodingKeys: String, CodingKey {
        case status
        case workspace
        case scannedFiles = "scanned_files"
        case indexedFiles = "indexed_files"
        case unchangedFiles = "unchanged_files"
        case removedFiles = "removed_files"
        case skippedFiles = "skipped_files"
    }
}

public struct WorkspaceCitation: Decodable, Hashable, Sendable {
    public let path: String
    public let startLine: Int
    public let endLine: Int
    public let score: Double

    public init(path: String, startLine: Int, endLine: Int, score: Double) {
        self.path = path
        self.startLine = startLine
        self.endLine = endLine
        self.score = score
    }

    enum CodingKeys: String, CodingKey {
        case path
        case startLine = "start_line"
        case endLine = "end_line"
        case score
    }
}

public struct WorkspaceResult: Decodable, Sendable {
    public let id: String
    public let name: String
    public let revision: String?
    public let retrievedChunks: Int
    public let truncated: Bool
    public let citations: [WorkspaceCitation]

    public init(
        id: String,
        name: String,
        revision: String?,
        retrievedChunks: Int,
        truncated: Bool,
        citations: [WorkspaceCitation]
    ) {
        self.id = id
        self.name = name
        self.revision = revision
        self.retrievedChunks = retrievedChunks
        self.truncated = truncated
        self.citations = citations
    }

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case revision
        case retrievedChunks = "retrieved_chunks"
        case truncated
        case citations
    }
}

public struct ChatRequest: Encodable, Sendable {
    public let requestID: String
    public let model: String
    public let messages: [APIChatMessage]
    public let context: [String]
    public let stream = true
    public let keepAlive = "forever"
    public let options: Options
    public let workspaceID: String?

    public init(
        requestID: String,
        model: String,
        messages: [APIChatMessage],
        context: [String],
        options: Options,
        workspaceID: String? = nil
    ) {
        self.requestID = requestID
        self.model = model
        self.messages = messages
        self.context = context
        self.options = options
        self.workspaceID = workspaceID
    }

    public struct Options: Encodable, Sendable {
        public var maxTokens: Int
        public var temperature: Double
        public var affinityKey: String?

        public init(maxTokens: Int, temperature: Double, affinityKey: String?) {
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.affinityKey = affinityKey
        }

        enum CodingKeys: String, CodingKey {
            case maxTokens = "num_predict"
            case temperature
            case affinityKey = "affinity_key"
        }
    }

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case model
        case messages
        case context
        case stream
        case keepAlive = "keep_alive"
        case options
        case workspaceID = "workspace_id"
    }
}

public struct ChatEvent: Decodable, Sendable {
    public struct Message: Decodable, Sendable {
        public let role: String?
        public let content: String

        public init(role: String?, content: String) {
            self.role = role
            self.content = content
        }
    }

    public struct MachBoost: Decodable, Sendable {
        public let backend: String?
        public let stats: GenerationStats?
        public let timeToFirstTokenSeconds: Double?
        public let workspace: WorkspaceResult?

        public init(
            backend: String?,
            stats: GenerationStats?,
            timeToFirstTokenSeconds: Double?,
            workspace: WorkspaceResult? = nil
        ) {
            self.backend = backend
            self.stats = stats
            self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
            self.workspace = workspace
        }

        enum CodingKeys: String, CodingKey {
            case backend
            case stats
            case timeToFirstTokenSeconds = "time_to_first_token_seconds"
            case workspace
        }
    }

    public let requestID: String?
    public let message: Message?
    public let done: Bool
    public let doneReason: String?
    public let totalDuration: Int64?
    public let evalDuration: Int64?
    public let evalCount: Int?
    public let machboost: MachBoost?
    public let error: String?

    public init(
        requestID: String?,
        message: Message?,
        done: Bool,
        doneReason: String?,
        totalDuration: Int64?,
        evalDuration: Int64?,
        evalCount: Int?,
        machboost: MachBoost?,
        error: String?
    ) {
        self.requestID = requestID
        self.message = message
        self.done = done
        self.doneReason = doneReason
        self.totalDuration = totalDuration
        self.evalDuration = evalDuration
        self.evalCount = evalCount
        self.machboost = machboost
        self.error = error
    }

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case message
        case done
        case doneReason = "done_reason"
        case totalDuration = "total_duration"
        case evalDuration = "eval_duration"
        case evalCount = "eval_count"
        case machboost
        case error
    }
}

public struct GenerationStats: Decodable, Sendable {
    public let generatedTokens: Int?
    public let generationSeconds: Double?
    public let promptTokens: Int?

    public init(generatedTokens: Int?, generationSeconds: Double?, promptTokens: Int?) {
        self.generatedTokens = generatedTokens
        self.generationSeconds = generationSeconds
        self.promptTokens = promptTokens
    }

    enum CodingKeys: String, CodingKey {
        case generatedTokens = "generated_tokens"
        case generationSeconds = "generation_seconds"
        case promptTokens = "prompt_tokens"
    }
}

public struct PullEvent: Decodable, Sendable {
    public let requestID: String?
    public let status: String?
    public let file: String?
    public let completed: Int64?
    public let total: Int64?
    public let unit: String?
    public let done: Bool
    public let path: String?
    public let error: String?

    public init(
        requestID: String?,
        status: String?,
        file: String?,
        completed: Int64?,
        total: Int64?,
        unit: String?,
        done: Bool,
        path: String?,
        error: String?
    ) {
        self.requestID = requestID
        self.status = status
        self.file = file
        self.completed = completed
        self.total = total
        self.unit = unit
        self.done = done
        self.path = path
        self.error = error
    }

    enum CodingKeys: String, CodingKey {
        case requestID = "request_id"
        case status
        case file
        case completed
        case total
        case unit
        case done
        case path
        case error
    }
}

public struct ModelInstance: Decodable, Identifiable, Sendable {
    public struct Scheduler: Decodable, Sendable {
        public let replicas: Int
        public let activeRequests: Int
        public let queuedRequests: Int

        public init(replicas: Int, activeRequests: Int, queuedRequests: Int) {
            self.replicas = replicas
            self.activeRequests = activeRequests
            self.queuedRequests = queuedRequests
        }

        enum CodingKeys: String, CodingKey {
            case replicas
            case activeRequests = "active_requests"
            case queuedRequests = "queued_requests"
        }
    }

    public let model: String
    public let backend: String
    public let idleSeconds: Double
    public let keepAliveSeconds: Double
    public let requests: Int
    public let capabilities: [String]
    public let scheduler: Scheduler

    public init(
        model: String,
        backend: String,
        idleSeconds: Double,
        keepAliveSeconds: Double,
        requests: Int,
        capabilities: [String],
        scheduler: Scheduler
    ) {
        self.model = model
        self.backend = backend
        self.idleSeconds = idleSeconds
        self.keepAliveSeconds = keepAliveSeconds
        self.requests = requests
        self.capabilities = capabilities
        self.scheduler = scheduler
    }

    public var id: String { "\(model)-\(backend)" }

    enum CodingKeys: String, CodingKey {
        case model
        case backend
        case idleSeconds = "idle_seconds"
        case keepAliveSeconds = "keep_alive_seconds"
        case requests
        case capabilities
        case scheduler
    }
}

public struct ModelsResponse: Decodable, Sendable {
    public let models: [ModelInstance]
}

public struct ServerMetrics: Decodable, Sendable {
    public struct Operations: Decodable, Sendable {
        public struct Latency: Decodable, Sendable {
            public let p50: Double
            public let p95: Double

            public init(p50: Double, p95: Double) {
                self.p50 = p50
                self.p95 = p95
            }
        }

        public struct Totals: Decodable, Sendable {
            public let started: Int
            public let completed: Int
            public let cancelled: Int
            public let failed: Int
            public let generatedTokens: Int

            public init(
                started: Int,
                completed: Int,
                cancelled: Int,
                failed: Int,
                generatedTokens: Int
            ) {
                self.started = started
                self.completed = completed
                self.cancelled = cancelled
                self.failed = failed
                self.generatedTokens = generatedTokens
            }

            enum CodingKeys: String, CodingKey {
                case started
                case completed
                case cancelled
                case failed
                case generatedTokens = "generated_tokens"
            }
        }

        public let activeCount: Int
        public let totals: Totals
        public let latencySeconds: Latency
        public let generationTokensPerSecond: Double

        public init(
            activeCount: Int,
            totals: Totals,
            latencySeconds: Latency = .init(p50: 0, p95: 0),
            generationTokensPerSecond: Double
        ) {
            self.activeCount = activeCount
            self.totals = totals
            self.latencySeconds = latencySeconds
            self.generationTokensPerSecond = generationTokensPerSecond
        }

        enum CodingKeys: String, CodingKey {
            case activeCount = "active_count"
            case totals
            case latencySeconds = "latency_seconds"
            case generationTokensPerSecond = "generation_tokens_per_second"
        }
    }

    public struct Scheduler: Decodable, Sendable {
        public let activeRequests: Int
        public let queuedRequests: Int
        public let rejectedRequests: Int

        public init(activeRequests: Int, queuedRequests: Int, rejectedRequests: Int) {
            self.activeRequests = activeRequests
            self.queuedRequests = queuedRequests
            self.rejectedRequests = rejectedRequests
        }

        enum CodingKeys: String, CodingKey {
            case activeRequests = "active_requests"
            case queuedRequests = "queued_requests"
            case rejectedRequests = "rejected_requests"
        }
    }

    public struct ProcessInfo: Decodable, Sendable {
        public let peakResidentMemoryBytes: Int64

        public init(peakResidentMemoryBytes: Int64) {
            self.peakResidentMemoryBytes = peakResidentMemoryBytes
        }

        enum CodingKeys: String, CodingKey {
            case peakResidentMemoryBytes = "peak_resident_memory_bytes"
        }
    }

    public let schema: String
    public let operations: Operations
    public let models: [ModelInstance]
    public let scheduler: Scheduler
    public let process: ProcessInfo

    public init(
        schema: String,
        operations: Operations,
        models: [ModelInstance],
        scheduler: Scheduler,
        process: ProcessInfo
    ) {
        self.schema = schema
        self.operations = operations
        self.models = models
        self.scheduler = scheduler
        self.process = process
    }
}

public struct ModelPreflightResponse: Decodable, Sendable {
    public struct Preflight: Decodable, Sendable {
        public let model: String
        public let backend: String
        public let capabilities: [String]
        public let runtimeAvailable: Bool
        public let cached: Bool
        public let modelType: String?
        public let supported: Bool
        public let reason: String?

        public init(
            model: String,
            backend: String,
            capabilities: [String],
            runtimeAvailable: Bool,
            cached: Bool,
            modelType: String?,
            supported: Bool,
            reason: String?
        ) {
            self.model = model
            self.backend = backend
            self.capabilities = capabilities
            self.runtimeAvailable = runtimeAvailable
            self.cached = cached
            self.modelType = modelType
            self.supported = supported
            self.reason = reason
        }

        enum CodingKeys: String, CodingKey {
            case model
            case backend
            case capabilities
            case runtimeAvailable = "runtime_available"
            case cached
            case modelType = "model_type"
            case supported
            case reason
        }
    }

    public let preflight: Preflight
}
