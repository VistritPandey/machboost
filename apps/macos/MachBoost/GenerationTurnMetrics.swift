import Foundation

struct GenerationTurnMetrics: Equatable {
    private(set) var generatedTokens = 0
    private(set) var generationSeconds = 0.0
    private(set) var totalDurationSeconds = 0.0
    private(set) var timeToFirstTokenSeconds: Double?
    private(set) var wasCancelled = false
    private(set) var rounds = 0
    private(set) var inferenceSource: String?
    private(set) var providerID: String?
    private(set) var providerLatencySeconds: Double?
    private(set) var providerCostUSD: Double?

    mutating func absorb(_ event: ChatEvent) {
        guard event.done else { return }
        rounds += 1
        wasCancelled = wasCancelled || event.doneReason == "cancelled"

        let tokens = event.evalCount ?? event.machboost?.stats?.generatedTokens ?? 0
        let seconds: Double
        if let duration = event.evalDuration, duration > 0 {
            seconds = Double(duration) / 1_000_000_000
        } else {
            seconds = max(0, event.machboost?.stats?.generationSeconds ?? 0)
        }
        generatedTokens += max(0, tokens)
        generationSeconds += seconds

        if let duration = event.totalDuration, duration > 0 {
            totalDurationSeconds += Double(duration) / 1_000_000_000
        }
        if timeToFirstTokenSeconds == nil {
            timeToFirstTokenSeconds = event.machboost?.timeToFirstTokenSeconds
        }
        if let route = event.machboost?.route {
            recordInferenceSource(route.source)
            providerID = route.providerID ?? providerID
            if let latency = route.latencySeconds {
                providerLatencySeconds = (providerLatencySeconds ?? 0) + latency
            }
            if let cost = route.costUSD {
                providerCostUSD = (providerCostUSD ?? 0) + cost
            }
        } else if event.machboost?.backend != nil {
            recordInferenceSource("local")
        }
    }

    var tokensPerSecond: Double? {
        guard generatedTokens > 0, generationSeconds > 0 else { return nil }
        return Double(generatedTokens) / generationSeconds
    }

    func apply(to message: ChatMessage) {
        message.wasCancelled = message.wasCancelled || wasCancelled
        message.durationSeconds = totalDurationSeconds > 0 ? totalDurationSeconds : nil
        message.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        message.generatedTokens = generatedTokens > 0 ? generatedTokens : nil
        message.tokensPerSecond = tokensPerSecond
        message.inferenceSource = inferenceSource
        message.providerID = providerID
        message.providerLatencySeconds = providerLatencySeconds
        message.providerCostUSD = providerCostUSD
    }

    private mutating func recordInferenceSource(_ source: String) {
        guard !source.isEmpty else { return }
        if let inferenceSource, inferenceSource != source {
            self.inferenceSource = "mixed"
        } else {
            self.inferenceSource = source
        }
    }
}
