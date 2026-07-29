import Foundation
import Sparkle

@MainActor
final class UpdateController: ObservableObject {
    private let updaterController: SPUStandardUpdaterController?

    init(startingUpdater: Bool = true, publicKey: String? = nil) {
        let environment = ProcessInfo.processInfo.environment
        let resolvedPublicKey = publicKey
            ?? Bundle.main.object(forInfoDictionaryKey: "SUPublicEDKey") as? String
            ?? ""
        guard
            environment["MACHBOOST_TESTING"] != "1",
            environment["MACHBOOST_UI_TESTING"] != "1",
            !resolvedPublicKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
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
        updaterController != nil
    }

    func checkForUpdates() {
        updaterController?.checkForUpdates(nil)
    }

    var automaticallyChecksForUpdates: Bool {
        get { updaterController?.updater.automaticallyChecksForUpdates ?? false }
        set { updaterController?.updater.automaticallyChecksForUpdates = newValue }
    }
}
