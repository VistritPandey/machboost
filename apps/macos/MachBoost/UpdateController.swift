import AppKit
import Foundation
import Sparkle

@MainActor
final class UpdateController: ObservableObject {
    typealias LatestReleaseFetcher = @Sendable (URL) async throws -> String?

    private let updaterController: SPUStandardUpdaterController?
    private let releasesURL: URL
    private let latestReleaseURL: URL
    private let openRelease: (URL) -> Void
    private let defaults: UserDefaults
    private let currentVersion: String
    private let fetchLatestRelease: LatestReleaseFetcher

    @Published private(set) var latestCommunityVersion: String?
    @Published private(set) var communityCheckCompleted = false
    @Published private(set) var communityCheckFailed = false
    @Published private(set) var isChecking = false
    @Published private(set) var lastCheckedAt: Date?

    private static let automaticCommunityChecksKey =
        "MachBoostAutomaticallyChecksCommunityReleases"

    init(
        startingUpdater: Bool = true,
        publicKey: String? = nil,
        releasesURL: URL = URL(
            string: "https://github.com/VistritPandey/machboost/releases/latest"
        )!,
        latestReleaseURL: URL = URL(
            string: "https://api.github.com/repos/VistritPandey/machboost/releases/latest"
        )!,
        openRelease: @escaping (URL) -> Void = { _ = NSWorkspace.shared.open($0) },
        defaults: UserDefaults = .standard,
        currentVersion: String = Bundle.main.object(
            forInfoDictionaryKey: "CFBundleShortVersionString"
        ) as? String ?? "0.0.0",
        fetchLatestRelease: LatestReleaseFetcher? = nil
    ) {
        self.releasesURL = releasesURL
        self.latestReleaseURL = latestReleaseURL
        self.openRelease = openRelease
        self.defaults = defaults
        self.currentVersion = currentVersion
        self.fetchLatestRelease = fetchLatestRelease ?? Self.fetchReleaseTag
        let environment = ProcessInfo.processInfo.environment
        let resolvedPublicKey = publicKey
            ?? Bundle.main.object(forInfoDictionaryKey: "SUPublicEDKey") as? String
            ?? ""
        if
            environment["MACHBOOST_TESTING"] != "1",
            environment["MACHBOOST_UI_TESTING"] != "1",
            Self.isValidSparklePublicKey(resolvedPublicKey)
        {
            updaterController = SPUStandardUpdaterController(
                startingUpdater: startingUpdater,
                updaterDelegate: nil,
                userDriverDelegate: nil
            )
        } else {
            updaterController = nil
        }

        if startingUpdater,
           updaterController == nil,
           environment["MACHBOOST_TESTING"] != "1",
           environment["MACHBOOST_UI_TESTING"] != "1",
           automaticallyChecksForUpdates
        {
            Task { await checkCommunityRelease() }
        }
    }

    var isAvailable: Bool {
        true
    }

    var supportsAutomaticUpdates: Bool {
        true
    }

    var actionTitle: String {
        "Check Now"
    }

    var updateAvailable: Bool {
        guard let latestCommunityVersion else { return false }
        return Self.isVersion(latestCommunityVersion, newerThan: currentVersion)
    }

    var canDownloadUpdate: Bool {
        updaterController == nil && updateAvailable
    }

    var downloadTitle: String {
        guard let latestCommunityVersion else { return "Download Update" }
        return "Download \(latestCommunityVersion)"
    }

    var deliveryDescription: String {
        if updaterController != nil {
            return "Signed updates install through Sparkle"
        }
        if updateAvailable, let latestCommunityVersion {
            return "\(latestCommunityVersion) is available on GitHub; installation is manual"
        }
        if communityCheckFailed {
            return "Could not check GitHub Releases; installation remains manual"
        }
        if communityCheckCompleted {
            return "Up to date; community installation remains manual"
        }
        return "Checks GitHub Releases; community installation is manual"
    }

    func checkForUpdates() {
        if let updaterController {
            updaterController.checkForUpdates(nil)
        } else {
            Task { await checkCommunityRelease() }
        }
    }

    func downloadUpdate() {
        guard updaterController == nil, updateAvailable else { return }
        openRelease(releasesURL)
    }

    var automaticallyChecksForUpdates: Bool {
        get {
            if let updaterController {
                return updaterController.updater.automaticallyChecksForUpdates
            }
            guard defaults.object(forKey: Self.automaticCommunityChecksKey) != nil else {
                return true
            }
            return defaults.bool(forKey: Self.automaticCommunityChecksKey)
        }
        set {
            if let updaterController {
                updaterController.updater.automaticallyChecksForUpdates = newValue
                return
            }
            defaults.set(newValue, forKey: Self.automaticCommunityChecksKey)
            if newValue {
                Task { await checkCommunityRelease() }
            }
        }
    }

    func checkCommunityRelease() async {
        guard updaterController == nil else { return }
        isChecking = true
        defer {
            isChecking = false
            communityCheckCompleted = true
            lastCheckedAt = .now
        }
        do {
            latestCommunityVersion = try await fetchLatestRelease(latestReleaseURL)
            communityCheckFailed = false
        } catch {
            communityCheckFailed = true
        }
    }

    private static func isValidSparklePublicKey(_ value: String) -> Bool {
        let key = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard
            !key.isEmpty,
            !key.contains("$("),
            let decoded = Data(base64Encoded: key)
        else {
            return false
        }
        return decoded.count == 32
    }

    private static func isVersion(_ candidate: String, newerThan current: String) -> Bool {
        let candidate = candidate.trimmingCharacters(in: CharacterSet(charactersIn: "vV"))
        let current = current.trimmingCharacters(in: CharacterSet(charactersIn: "vV"))
        return candidate.compare(current, options: .numeric) == .orderedDescending
    }

    private static func fetchReleaseTag(from url: URL) async throws -> String? {
        var request = URLRequest(url: url)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("MachBoost-macOS", forHTTPHeaderField: "User-Agent")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode)
        else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(Release.self, from: data).tagName
    }

    private struct Release: Decodable {
        let tagName: String

        enum CodingKeys: String, CodingKey {
            case tagName = "tag_name"
        }
    }
}
