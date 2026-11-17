/**
 * ApproxMAC Client-Side Auth Guard
 * Blocks unauthenticated users from opening or viewing application pages.
 */
(function() {
  const token = localStorage.getItem("approxmac_token");
  const cookies = document.cookie || "";
  const hasCookie = cookies.includes("session_token=");

  // 1. Instant check: redirect immediately if no credentials exist
  if (!token && !hasCookie) {
    window.location.replace("login.html?error=login_required");
    return;
  }

  // 2. Validate session with the server
  fetch("/api/auth/me", {
    headers: token ? { "Authorization": "Bearer " + token } : {}
  })
    .then(function(res) {
      if (!res.ok) {
        localStorage.removeItem("approxmac_token");
        localStorage.removeItem("approxmac_employee");
        window.location.replace("login.html?error=session_expired");
        return null;
      }
      return res.json();
    })
    .then(function(data) {
      if (data && data.employee) {
        localStorage.setItem("approxmac_employee", JSON.stringify(data.employee));
        const el = document.getElementById("nav-user-name");
        if (el) {
          el.innerText = data.employee.name;
        }
        const empIdEl = document.getElementById("nav-user-id");
        if (empIdEl) {
          empIdEl.innerText = data.employee.employee_id;
        }
      }
    })
    .catch(function(err) {
      console.warn("Auth check warning:", err);
    });
})();

// Global logout function for all pages
function logoutEmployee() {
  fetch("/api/auth/logout", { method: "POST" })
    .finally(function() {
      localStorage.removeItem("approxmac_token");
      localStorage.removeItem("approxmac_employee");
      window.location.replace("login.html?msg=logged_out");
    });
}
