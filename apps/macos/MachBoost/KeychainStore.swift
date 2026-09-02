import Foundation
import Security

enum KeychainStore {
    private static let service = "io.machboost.MachBoost"
    private static let account = "lan-api-token"
    private static let communityStore = CommunityCredentialStore()
    private static let usesCommunityStore = signingTeamIdentifier() == nil

    static func token() -> String? {
        value(account: account)
    }

    static func providerSecret(id: String) -> String? {
        value(account: "provider-\(id)")
    }

    static func teamToken(profileID: UUID) -> String? {
        value(account: "team-host-\(profileID.uuidString.lowercased())")
    }

    static func providerSecretAsync(id: String) async -> String? {
        await Task.detached(priority: .userInitiated) {
            providerSecret(id: id)
        }.value
    }

    static func teamTokenAsync(profileID: UUID) async -> String? {
        await Task.detached(priority: .userInitiated) {
            teamToken(profileID: profileID)
        }.value
    }

    private static func value(account: String) -> String? {
        if usesCommunityStore {
            return communityStore.value(account: account)
        }
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecMatchLimit as String: kSecMatchLimitOne,
            kSecReturnData as String: true,
        ]
        var result: CFTypeRef?
        guard
            SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
            let data = result as? Data
        else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    @discardableResult
    static func generateToken() throws -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        let status = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        guard status == errSecSuccess else {
            throw KeychainError(status: status)
        }
        let token = Data(bytes).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
        try save(token: token)
        return token
    }

    static func tokenOrCreate() throws -> String {
        if let existing = token(), !existing.isEmpty {
            return existing
        }
        return try generateToken()
    }

    static func tokenOrCreateAsync() async throws -> String {
        try await Task.detached(priority: .userInitiated) {
            try tokenOrCreate()
        }.value
    }

    static func generateTokenAsync() async throws -> String {
        try await Task.detached(priority: .userInitiated) {
            try generateToken()
        }.value
    }

    static func save(token: String) throws {
        try save(value: token, account: account)
    }

    static func saveProviderSecret(_ secret: String, id: String) throws {
        try save(value: secret, account: "provider-\(id)")
    }

    static func saveProviderSecretAsync(_ secret: String, id: String) async throws {
        try await Task.detached(priority: .userInitiated) {
            try saveProviderSecret(secret, id: id)
        }.value
    }

    static func saveTeamToken(_ token: String, profileID: UUID) throws {
        try save(
            value: token,
            account: "team-host-\(profileID.uuidString.lowercased())"
        )
    }

    static func saveTeamTokenAsync(_ token: String, profileID: UUID) async throws {
        try await Task.detached(priority: .userInitiated) {
            try saveTeamToken(token, profileID: profileID)
        }.value
    }

    static func deleteProviderSecret(id: String) throws {
        try delete(account: "provider-\(id)")
    }

    static func deleteProviderSecretAsync(id: String) async throws {
        try await Task.detached(priority: .userInitiated) {
            try deleteProviderSecret(id: id)
        }.value
    }

    static func deleteTeamToken(profileID: UUID) throws {
        try delete(account: "team-host-\(profileID.uuidString.lowercased())")
    }

    static func deleteTeamTokenAsync(profileID: UUID) async throws {
        try await Task.detached(priority: .userInitiated) {
            try deleteTeamToken(profileID: profileID)
        }.value
    }

    private static func delete(account: String) throws {
        if usesCommunityStore {
            try communityStore.delete(account: account)
            return
        }
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError(status: status)
        }
    }

    private static func save(value: String, account: String) throws {
        if usesCommunityStore {
            try communityStore.save(value: value, account: account)
            return
        }
        let base: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let data = Data(value.utf8)
        let updateStatus = SecItemUpdate(
            base as CFDictionary,
            [kSecValueData as String: data] as CFDictionary
        )
        if updateStatus == errSecSuccess {
            return
        }
        guard updateStatus == errSecItemNotFound else {
            throw KeychainError(status: updateStatus)
        }
        var create = base
        create[kSecValueData as String] = data
        let createStatus = SecItemAdd(create as CFDictionary, nil)
        guard createStatus == errSecSuccess else {
            throw KeychainError(status: createStatus)
        }
    }

    private static func signingTeamIdentifier() -> String? {
        var code: SecCode?
        guard SecCodeCopySelf([], &code) == errSecSuccess, let code else {
            return nil
        }
        var staticCode: SecStaticCode?
        guard
            SecCodeCopyStaticCode(code, [], &staticCode) == errSecSuccess,
            let staticCode
        else {
            return nil
        }
        var information: CFDictionary?
        guard
            SecCodeCopySigningInformation(
                staticCode,
                SecCSFlags(rawValue: kSecCSSigningInformation),
                &information
            ) == errSecSuccess,
            let values = information as? [String: Any]
        else {
            return nil
        }
        return values[kSecCodeInfoTeamIdentifier as String] as? String
    }
}

final class CommunityCredentialStore: @unchecked Sendable {
    let credentialsURL: URL

    private let root: URL
    private let lock = NSLock()

    init(root: URL? = nil) {
        let resolvedRoot = root ?? FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MachBoost", isDirectory: true)
        self.root = resolvedRoot
        credentialsURL = resolvedRoot.appendingPathComponent(
            "credentials.community.json",
            isDirectory: false
        )
    }

    func value(account: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return load()[account]
    }

    func save(value: String, account: String) throws {
        lock.lock()
        defer { lock.unlock() }
        var values = load()
        values[account] = value
        try persist(values)
    }

    func delete(account: String) throws {
        lock.lock()
        defer { lock.unlock() }
        var values = load()
        guard values.removeValue(forKey: account) != nil else { return }
        try persist(values)
    }

    private func load() -> [String: String] {
        guard
            let data = try? Data(contentsOf: credentialsURL),
            let values = try? JSONDecoder().decode([String: String].self, from: data)
        else {
            return [:]
        }
        return values
    }

    private func persist(_ values: [String: String]) throws {
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        try fileManager.setAttributes(
            [.posixPermissions: NSNumber(value: 0o700)],
            ofItemAtPath: root.path
        )
        let data = try JSONEncoder().encode(values)
        try data.write(to: credentialsURL, options: .atomic)
        try fileManager.setAttributes(
            [.posixPermissions: NSNumber(value: 0o600)],
            ofItemAtPath: credentialsURL.path
        )
    }
}

struct KeychainError: LocalizedError {
    let status: OSStatus

    var errorDescription: String? {
        SecCopyErrorMessageString(status, nil) as String?
            ?? "Keychain operation failed (\(status))."
    }
}
