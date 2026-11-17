/**
 * ApproxMAC Cinematic Page Transitions & Mobile Navigation Engine
 * 1. Ultra-smooth physics-based animation between pages (randomly top-to-center or bottom-to-top).
 * 2. Mobile 2-dots toggle button on top right with smooth left-to-right sliding drawer navigation.
 * 3. Dynamic active nav link auto-detection.
 */
(function() {
  if (window.__approxmacCinematicLoaded) return;
  window.__approxmacCinematicLoaded = true;

  // Determine entry direction: randomly chooses "top" (top-to-center) or "bottom" (bottom-to-top)
  function getEntranceDirection() {
    let dir = sessionStorage.getItem("cinematic_dir");
    if (!dir) {
      dir = Math.random() < 0.5 ? "top" : "bottom";
    }
    sessionStorage.removeItem("cinematic_dir");
    return dir;
  }

  function getTarget() {
    return document.querySelector(".main-wrapper") || 
           document.querySelector(".portal-wrapper") || 
           document.querySelector("main") ||
           document.querySelector(".portal-container") ||
           document.body;
  }

  function applyEntrance() {
    const dir = getEntranceDirection();
    const target = getTarget();
    if (target) {
      target.classList.remove("cinematic-travel-top", "cinematic-travel-bottom", "cinematic-exit-up", "cinematic-exit-down");
      void target.offsetWidth; // Force CSS reflow for clean restart
      target.classList.add(dir === "top" ? "cinematic-travel-top" : "cinematic-travel-bottom");
    }
  }

  // Mobile Navigation: Two-dots toggle button on top-right & slow cinematic sliding drawer
  function initMobileNav() {
    const sidebar = document.querySelector(".sidebar");
    if (!sidebar) return;

    // 1. Two-dots button on top right
    let toggleBtn = document.getElementById("mobileNavToggle");
    if (!toggleBtn) {
      toggleBtn = document.createElement("button");
      toggleBtn.id = "mobileNavToggle";
      toggleBtn.className = "mobile-dots-btn";
      toggleBtn.setAttribute("aria-label", "Toggle navigation menu");
      toggleBtn.setAttribute("title", "Toggle Navigation");
      toggleBtn.innerHTML = '<span class="mobile-dot"></span><span class="mobile-dot"></span>';
      document.body.appendChild(toggleBtn);
    }

    // 2. Backdrop overlay
    let backdrop = document.getElementById("sidebarBackdrop");
    if (!backdrop) {
      backdrop = document.createElement("div");
      backdrop.id = "sidebarBackdrop";
      backdrop.className = "sidebar-backdrop";
      document.body.appendChild(backdrop);
    }

    // 3. Sidebar close button inside brand/header
    let closeBtn = sidebar.querySelector(".sidebar-close-btn");
    if (!closeBtn) {
      closeBtn = document.createElement("button");
      closeBtn.className = "sidebar-close-btn";
      closeBtn.setAttribute("aria-label", "Close Menu");
      closeBtn.innerHTML = "&times;";
      const brand = sidebar.querySelector(".brand") || sidebar.firstElementChild;
      if (brand) {
        brand.style.position = "relative";
        brand.appendChild(closeBtn);
      }
    }

    function openNav() {
      sidebar.classList.add("mobile-open");
      backdrop.classList.add("active");
      toggleBtn.classList.add("active");
      document.body.classList.add("mobile-menu-active");
    }

    function closeNav() {
      sidebar.classList.remove("mobile-open");
      backdrop.classList.remove("active");
      toggleBtn.classList.remove("active");
      document.body.classList.remove("mobile-menu-active");
    }

    // Ensure strictly closed on mobile view initially
    closeNav();

    toggleBtn.addEventListener("click", function(e) {
      e.stopPropagation();
      e.preventDefault();
      if (sidebar.classList.contains("mobile-open")) {
        closeNav();
      } else {
        openNav();
      }
    });

    if (closeBtn) {
      closeBtn.addEventListener("click", function(e) {
        e.stopPropagation();
        e.preventDefault();
        closeNav();
      });
    }

    backdrop.addEventListener("click", function(e) {
      e.stopPropagation();
      closeNav();
    });

    // Close when clicking nav links on mobile
    sidebar.querySelectorAll(".nav-link").forEach(function(link) {
      link.addEventListener("click", function() {
        closeNav();
      });
    });

    // Close on Escape key
    document.addEventListener("keydown", function(e) {
      if (e.key === "Escape" && sidebar.classList.contains("mobile-open")) {
        closeNav();
      }
    });

    // 4. Ensure active page link is marked accurately
    const currentPath = (window.location.pathname.split("/").pop() || "index.html").split("?")[0].split("#")[0];
    sidebar.querySelectorAll(".nav-link").forEach(function(link) {
      const href = (link.getAttribute("href") || "").split("?")[0].split("#")[0];
      if (href && href === currentPath) {
        link.classList.add("active");
      } else if (href && !href.startsWith("#") && href.endsWith(".html")) {
        link.classList.remove("active");
      }
    });

    // Also highlight active footer link if present
    document.querySelectorAll(".footer-link").forEach(function(link) {
      const href = (link.getAttribute("href") || "").split("?")[0].split("#")[0];
      if (href && href === currentPath) {
        link.classList.add("active");
      } else if (href && !href.startsWith("#") && href.endsWith(".html")) {
        link.classList.remove("active");
      }
    });
  }

  // Run on page load
  function init() {
    applyEntrance();
    initMobileNav();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Handle browser back/forward cache restoration
  window.addEventListener("pageshow", function(e) {
    if (e.persisted) {
      applyEntrance();
    }
  });

  // Intercept navigation links for smooth cinematic exits with event delegation
  document.addEventListener("click", function(e) {
    const link = e.target && e.target.closest ? e.target.closest("a[href]") : null;
    if (!link) return;

    const href = link.getAttribute("href");
    if (
      !href ||
      href.startsWith("#") ||
      href.startsWith("javascript:") ||
      href.startsWith("mailto:") ||
      href.startsWith("tel:") ||
      link.target === "_blank" ||
      link.hasAttribute("download") ||
      href.includes("/api/")
    ) {
      return;
    }

    if (e.ctrlKey || e.metaKey || e.shiftKey || e.defaultPrevented) return;

    // Check if clicking current page URL
    const currentFile = (window.location.pathname.split("/").pop() || "index.html").split("?")[0].split("#")[0];
    const targetFile = href.split("?")[0].split("#")[0].split("/").pop() || "index.html";
    if (currentFile === targetFile && !href.includes("?")) {
      return;
    }

    e.preventDefault();

    // Randomly pick next page entrance direction: "top" or "bottom"
    const nextDir = Math.random() < 0.5 ? "top" : "bottom";
    sessionStorage.setItem("cinematic_dir", nextDir);

    // Smooth cinematic exit on current view
    const target = getTarget();
    if (target) {
      target.classList.remove("cinematic-travel-top", "cinematic-travel-bottom");
      target.classList.add(nextDir === "top" ? "cinematic-exit-down" : "cinematic-exit-up");
    }

    // Navigate to new page smoothly
    setTimeout(function() {
      window.location.href = href;
    }, 360);
  }, true);
})();
