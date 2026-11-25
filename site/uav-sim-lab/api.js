const base = window.LAB_CONFIG.apiBase.replace(/\/$/, "");
let accessCode = sessionStorage.getItem("uav-lab-access") || "";

export function rememberAccess(code) {
  accessCode = code;
  if (code) sessionStorage.setItem("uav-lab-access", code);
  else sessionStorage.removeItem("uav-lab-access");
}

export function hasAccess() { return Boolean(accessCode); }

export async function request(path, {method = "GET", body, timeout = 120000, key} = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  const headers = {};
  const code = key === undefined ? accessCode : key;
  if (code) headers["X-Lab-Key"] = code;
  if (body && !(body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(body);
  }
  try {
    const response = await fetch(base + path, {method, headers, body, signal: controller.signal, credentials: "omit"});
    const result = await response.json();
    if (!response.ok) {
      let message = result.detail;
      if (Array.isArray(message)) message = message.map(e => `${e.loc.at(-1)}: ${e.msg}`).join("; ");
      const error = new Error(typeof message === "string" ? message : "The request could not be completed.");
      error.status = response.status;
      throw error;
    }
    return result;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("Compute took too long to respond. Please retry; a cold GPU can take a moment to start.");
    if (error instanceof TypeError) throw new Error("Could not reach lab compute. Check your connection and try again.");
    throw error;
  } finally { clearTimeout(timer); }
}
