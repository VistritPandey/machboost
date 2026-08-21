@preconcurrency import Foundation
import Observation

struct DiscoveredMachBoostHost: Identifiable, Equatable, Sendable {
    let id: String
    let name: String
    let endpoint: URL
}

@MainActor
@Observable
final class MachBoostHostDiscovery: NSObject,
    @preconcurrency NetServiceBrowserDelegate,
    @preconcurrency NetServiceDelegate
{
    private let browser = NetServiceBrowser()
    private var services: [String: NetService] = [:]
    private var publisher: NetService?
    private(set) var hosts: [DiscoveredMachBoostHost] = []

    override init() {
        super.init()
        browser.delegate = self
    }

    func start() {
        browser.searchForServices(ofType: "_machboost._tcp.", inDomain: "local.")
    }

    func stop() {
        browser.stop()
        services.values.forEach { $0.stop() }
        services.removeAll()
        hosts = []
        stopPublishing()
    }

    func publish(name: String, port: Int) {
        stopPublishing()
        let service = NetService(
            domain: "local.",
            type: "_machboost._tcp.",
            name: name,
            port: Int32(port)
        )
        service.setTXTRecord(
            NetService.data(fromTXTRecord: [
                "version": Data((Bundle.main.object(
                    forInfoDictionaryKey: "CFBundleShortVersionString"
                ) as? String ?? "development").utf8),
                "path": Data("/".utf8),
            ])
        )
        service.publish()
        publisher = service
    }

    func stopPublishing() {
        publisher?.stop()
        publisher = nil
    }

    func netServiceBrowser(
        _ browser: NetServiceBrowser,
        didFind service: NetService,
        moreComing: Bool
    ) {
        let key = serviceKey(service)
        services[key] = service
        service.delegate = self
        service.resolve(withTimeout: 4)
    }

    func netServiceBrowser(
        _ browser: NetServiceBrowser,
        didRemove service: NetService,
        moreComing: Bool
    ) {
        let key = serviceKey(service)
        services.removeValue(forKey: key)
        hosts.removeAll { $0.id == key }
    }

    func netServiceDidResolveAddress(_ sender: NetService) {
        guard let endpoint = endpoint(for: sender) else { return }
        let key = serviceKey(sender)
        let host = DiscoveredMachBoostHost(
            id: key,
            name: sender.name,
            endpoint: endpoint
        )
        hosts.removeAll { $0.id == key }
        hosts.append(host)
        hosts.sort { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    private func serviceKey(_ service: NetService) -> String {
        "\(service.name).\(service.type)\(service.domain)"
    }

    private func endpoint(for service: NetService) -> URL? {
        guard service.port > 0 else { return nil }
        let rawHost = (service.hostName ?? "").trimmingCharacters(in: CharacterSet(charactersIn: "."))
        guard !rawHost.isEmpty else { return nil }
        var components = URLComponents()
        components.scheme = "http"
        components.host = rawHost
        components.port = service.port
        return components.url
    }
}
