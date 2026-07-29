const repository = "VistritPandey/machboost";
const latestReleaseURL = `https://github.com/${repository}/releases/latest`;

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "DMG";
  }

  const units = ["B", "KB", "MB", "GB"];
  const unitIndex = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** unitIndex;
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIndex]}`;
}

function initializeIcons() {
  if (window.lucide) {
    window.lucide.createIcons({
      attrs: {
        "aria-hidden": "true",
        "stroke-width": "2",
      },
    });
  }
}

function initializeMobileNavigation() {
  const button = document.querySelector(".mobile-menu");
  const navigation = document.querySelector("#mobile-nav");

  if (!button || !navigation) {
    return;
  }

  button.addEventListener("click", () => {
    const isOpen = button.getAttribute("aria-expanded") === "true";
    button.setAttribute("aria-expanded", String(!isOpen));
    navigation.hidden = isOpen;

    const icon = button.querySelector("svg");
    if (icon) {
      icon.setAttribute("data-lucide", isOpen ? "menu" : "x");
      initializeIcons();
    }
  });

  navigation.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      button.setAttribute("aria-expanded", "false");
      navigation.hidden = true;
    });
  });
}

async function initializeRelease() {
  const downloadButtons = [
    document.querySelector("#download-button"),
    document.querySelector("#install-download-button"),
  ].filter(Boolean);
  const releaseStatus = document.querySelector("#release-status");
  const releaseVersion = document.querySelector("#release-version");
  const releaseSize = document.querySelector("#release-size");
  const checksumLink = document.querySelector(".checksum-link");

  downloadButtons.forEach((button) => {
    button.href = latestReleaseURL;
  });

  try {
    const response = await fetch(`https://api.github.com/repos/${repository}/releases/latest`, {
      headers: {
        Accept: "application/vnd.github+json",
      },
    });

    if (!response.ok) {
      throw new Error(`GitHub returned ${response.status}`);
    }

    const release = await response.json();
    const dmg = release.assets?.find((asset) => asset.name.toLowerCase().endsWith(".dmg"));

    if (!dmg) {
      throw new Error("The latest release does not contain a DMG");
    }

    downloadButtons.forEach((button) => {
      button.href = dmg.browser_download_url;
    });

    const published = release.published_at
      ? new Intl.DateTimeFormat("en", {
          month: "short",
          day: "numeric",
          year: "numeric",
        }).format(new Date(release.published_at))
      : null;

    if (releaseStatus) {
      releaseStatus.textContent = [
        release.tag_name,
        published,
        "unsigned community preview",
        "arm64",
        "macOS 14+",
      ]
        .filter(Boolean)
        .join(" · ");
    }
    if (releaseVersion) {
      releaseVersion.textContent = release.tag_name;
    }
    if (releaseSize) {
      releaseSize.textContent = formatBytes(dmg.size);
    }
    if (checksumLink) {
      checksumLink.href = release.html_url;
    }
  } catch (error) {
    if (releaseStatus) {
      releaseStatus.textContent =
        "Unsigned community preview · arm64 · macOS 14+ · open GitHub for availability";
    }
  }
}

function initializeCopyButtons() {
  document.querySelectorAll(".copy-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const targetID = button.dataset.copyTarget;
      const target = targetID ? document.getElementById(targetID) : null;
      if (!target) {
        return;
      }

      try {
        await navigator.clipboard.writeText(target.innerText);
        button.setAttribute("aria-label", "Copied");
        button.innerHTML =
          '<i data-lucide="check" aria-hidden="true"></i><span class="sr-only">Copied</span>';
        initializeIcons();

        window.setTimeout(() => {
          button.setAttribute("aria-label", "Copy example");
          button.innerHTML =
            '<i data-lucide="copy" aria-hidden="true"></i><span class="sr-only">Copy example</span>';
          initializeIcons();
        }, 1800);
      } catch (error) {
        button.setAttribute("aria-label", "Copy failed");
      }
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initializeIcons();
  initializeMobileNavigation();
  initializeCopyButtons();
  initializeRelease();
});
