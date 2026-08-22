@preconcurrency import Foundation
import Observation

struct DiscoveredMachBoostHost: Identifiable, Equatable, Sendable {
    let id: String
    let name: String
    let endpoint: URL
    let deviceID: String?
    let version: String?

    init(
        id: String,
        name: String,
        endpoint: URL,
        deviceID: String? = nil,
        version: String? = nil
    ) {
        self.id = id
        self.name = name
        self.endpoint = endpoint
        self.deviceID = deviceID
        self.version = version
    }
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
    private var localDeviceID: String?
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

    func publish(name: String, port: Int, deviceID: String? = nil) {
        stopPublishing()
        localDeviceID = deviceID
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
                "device_id": Data((deviceID ?? "").utf8),
            ])
        )
        service.publish()
        publisher = service
    }

    func stopPublishing() {
        publisher?.stop()
        publisher = nil
        localDeviceID = nil
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
        let metadata = serviceMetadata(sender)
        guard !Self.isSelf(deviceID: metadata.deviceID, localDeviceID: localDeviceID) else {
            services.removeValue(forKey: key)
            hosts.removeAll { $0.id == key }
            sender.stop()
            return
        }
        let host = DiscoveredMachBoostHost(
            id: key,
            name: sender.name,
            endpoint: endpoint,
            deviceID: metadata.deviceID,
            version: metadata.version
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

    private func serviceMetadata(_ service: NetService) -> (deviceID: String?, version: String?) {
        guard let data = service.txtRecordData() else { return (nil, nil) }
        let record = NetService.dictionary(fromTXTRecord: data)
        return (
            decodedTXTValue(record["device_id"]),
            decodedTXTValue(record["version"])
        )
    }

    private func decodedTXTValue(_ data: Data?) -> String? {
        guard
            let data,
            let value = String(data: data, encoding: .utf8),
            !value.isEmpty
        else { return nil }
        return value
    }

    static func isSelf(deviceID: String?, localDeviceID: String?) -> Bool {
        guard let deviceID, let localDeviceID else { return false }
        return deviceID == localDeviceID
    }
}
