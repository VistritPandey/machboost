import XCTest

final class MachBoostUITests: XCTestCase {
    @MainActor
    func testPrimaryWorkspaceAndNavigationAreReachable() {
        let app = launchApp()
        XCTAssertTrue(app.staticTexts["MachBoost"].waitForExistence(timeout: 10))

        if app.buttons["Continue"].exists {
            app.buttons["Continue"].click()
        }

        XCTAssertTrue(app.staticTexts["Chats"].exists)
        XCTAssertTrue(app.staticTexts["Models"].exists)
        XCTAssertTrue(app.staticTexts["Server"].exists)
        XCTAssertTrue(app.staticTexts["Settings"].exists)
    }

    @MainActor
    func testServerDeveloperSurfaceOpens() {
        let app = launchApp()
        XCTAssertTrue(app.staticTexts["Server"].waitForExistence(timeout: 10))
        app.staticTexts["Server"].click()
        XCTAssertTrue(app.buttons["Developer"].waitForExistence(timeout: 3))
        app.buttons["Developer"].click()
        XCTAssertTrue(app.staticTexts["Endpoint"].exists)
        XCTAssertTrue(app.staticTexts["OpenAI Python"].exists)
    }

    @MainActor
    private func launchApp() -> XCUIApplication {
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launchEnvironment["MACHBOOST_SOURCE_ROOT"] = repositoryRoot()
        app.launch()
        return app
    }

    nonisolated private func repositoryRoot() -> String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .path
    }
}
