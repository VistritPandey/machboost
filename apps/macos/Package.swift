// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "MachBoostDesktop",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "MachBoost", targets: ["MachBoost"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/sparkle-project/Sparkle",
            exact: "2.9.4"
        ),
    ],
    targets: [
        .executableTarget(
            name: "MachBoost",
            dependencies: [
                .product(name: "Sparkle", package: "Sparkle"),
            ],
            path: "MachBoost",
            exclude: ["Info.plist", "MachBoost.entitlements"]
        ),
        .testTarget(
            name: "MachBoostTests",
            dependencies: ["MachBoost"],
            path: "MachBoostTests"
        ),
    ]
)
