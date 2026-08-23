import Foundation
import Security

enum KeychainStore {
    private static let service = "io.machboost.MachBoost"
    private static let account = "lan-api-token"

    static func token() -> String? {
        value(account: account)
    }

    static func tokenAsync() async -> String? {
        await Task.detached(priority: .userInitiated) {
            token()
        }.value
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
        if let current = storedValue(account: account, service: service) {
            return current
        }

        // Service names can change when an app moves from a development identity
        // to its public bundle identity. Recover the same account without baking a
        // historical identifier into the release, then migrate it forward.
        guard let migrated = storedValue(account: account, service: nil) else {
            return nil
        }
        try? save(value: migrated, account: account)
        return migrated
    }

    private static func storedValue(account: String, service: String?) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: account,
            kSecMatchLimit as String: kSecMatchLimitOne,
            kSecReturnData as String: true,
        ]
        var scopedQuery = query
        if let service {
            scopedQuery[kSecAttrService as String] = service
        }
        var result: CFTypeRef?
        guard
            SecItemCopyMatching(scopedQuery as CFDictionary, &result) == errSecSuccess,
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
}

struct KeychainError: LocalizedError {
    let status: OSStatus

    var errorDescription: String? {
        SecCopyErrorMessageString(status, nil) as String?
            ?? "Keychain operation failed (\(status))."
    }
}
