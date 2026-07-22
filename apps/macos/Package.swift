// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "MachBoostDesktop",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "MachBoost", targets: ["MachBoost"]),
        .library(name: "MachBoostDaemonClient", targets: ["MachBoostDaemonClient"]),
        .library(name: "MachBoostPersistence", targets: ["MachBoostPersistence"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/sparkle-project/Sparkle",
            exact: "2.9.4"
        ),
    ],
    targets: [
        .target(
            name: "MachBoostDaemonClient",
            path: "MachBoostDaemonClient"
        ),
        .target(
            name: "MachBoostPersistence",
            path: "MachBoostPersistence"
        ),
        .executableTarget(
            name: "MachBoost",
            dependencies: [
                "MachBoostDaemonClient",
                "MachBoostPersistence",
                .product(name: "Sparkle", package: "Sparkle"),
            ],
            path: "MachBoost",
            exclude: ["Info.plist", "MachBoost.entitlements"]
        ),
        .testTarget(
            name: "MachBoostTests",
            dependencies: [
                "MachBoost",
                "MachBoostDaemonClient",
                "MachBoostPersistence",
            ],
            path: "MachBoostTests"
        ),
    ]
)
