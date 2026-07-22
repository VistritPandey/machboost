import XCTest

@MainActor
final class MachBoostUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchEnvironment["MACHBOOST_SOURCE_ROOT"] = repositoryRoot()
        app.launch()
    }

    func testPrimaryWorkspaceAndNavigationAreReachable() {
        XCTAssertTrue(app.staticTexts["MachBoost"].waitForExistence(timeout: 10))

        if app.buttons["Continue"].exists {
            app.buttons["Continue"].click()
        }

        XCTAssertTrue(app.staticTexts["Chats"].exists)
        XCTAssertTrue(app.staticTexts["Models"].exists)
        XCTAssertTrue(app.staticTexts["Server"].exists)
        XCTAssertTrue(app.staticTexts["Settings"].exists)
    }

    func testServerDeveloperSurfaceOpens() {
        XCTAssertTrue(app.staticTexts["Server"].waitForExistence(timeout: 10))
        app.staticTexts["Server"].click()
        XCTAssertTrue(app.buttons["Developer"].waitForExistence(timeout: 3))
        app.buttons["Developer"].click()
        XCTAssertTrue(app.staticTexts["Endpoint"].exists)
        XCTAssertTrue(app.staticTexts["OpenAI Python"].exists)
    }

    private func repositoryRoot() -> String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .path
    }
}
