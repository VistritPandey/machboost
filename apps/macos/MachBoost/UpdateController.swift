import AppKit
import Foundation
import Sparkle

@MainActor
final class UpdateController: ObservableObject {
    private let updaterController: SPUStandardUpdaterController?
    private let releasesURL: URL
    private let openRelease: (URL) -> Void

    init(
        startingUpdater: Bool = true,
        publicKey: String? = nil,
        releasesURL: URL = URL(
            string: "https://github.com/VistritPandey/machboost/releases/latest"
        )!,
        openRelease: @escaping (URL) -> Void = { _ = NSWorkspace.shared.open($0) }
    ) {
        self.releasesURL = releasesURL
        self.openRelease = openRelease
        let environment = ProcessInfo.processInfo.environment
        let resolvedPublicKey = publicKey
            ?? Bundle.main.object(forInfoDictionaryKey: "SUPublicEDKey") as? String
            ?? ""
        guard
            environment["MACHBOOST_TESTING"] != "1",
            environment["MACHBOOST_UI_TESTING"] != "1",
            Self.isValidSparklePublicKey(resolvedPublicKey)
        else {
            updaterController = nil
            return
        }

        updaterController = SPUStandardUpdaterController(
            startingUpdater: startingUpdater,
            updaterDelegate: nil,
            userDriverDelegate: nil
        )
    }

    var isAvailable: Bool {
        true
    }

    var supportsAutomaticUpdates: Bool {
        updaterController != nil
    }

    var actionTitle: String {
        supportsAutomaticUpdates ? "Check for updates" : "View latest release"
    }

    var deliveryDescription: String {
        supportsAutomaticUpdates
            ? "Signed updates install through Sparkle"
            : "Community builds update through GitHub Releases"
    }

    func checkForUpdates() {
        if let updaterController {
            updaterController.checkForUpdates(nil)
        } else {
            openRelease(releasesURL)
        }
    }

    var automaticallyChecksForUpdates: Bool {
        get { updaterController?.updater.automaticallyChecksForUpdates ?? false }
        set { updaterController?.updater.automaticallyChecksForUpdates = newValue }
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
}
