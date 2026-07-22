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

public struct ChatRequest: Encodable, Sendable {
    public let requestID: String
    public let model: String
    public let messages: [APIChatMessage]
    public let context: [String]
    public let stream = true
    public let keepAlive = "forever"
    public let options: Options

    public init(
        requestID: String,
        model: String,
        messages: [APIChatMessage],
        context: [String],
        options: Options
    ) {
        self.requestID = requestID
        self.model = model
        self.messages = messages
        self.context = context
        self.options = options
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

        public init(
            backend: String?,
            stats: GenerationStats?,
            timeToFirstTokenSeconds: Double?
        ) {
            self.backend = backend
            self.stats = stats
            self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        }

        enum CodingKeys: String, CodingKey {
            case backend
            case stats
            case timeToFirstTokenSeconds = "time_to_first_token_seconds"
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
        public let generationTokensPerSecond: Double

        public init(
            activeCount: Int,
            totals: Totals,
            generationTokensPerSecond: Double
        ) {
            self.activeCount = activeCount
            self.totals = totals
            self.generationTokensPerSecond = generationTokensPerSecond
        }

        enum CodingKeys: String, CodingKey {
            case activeCount = "active_count"
            case totals
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
