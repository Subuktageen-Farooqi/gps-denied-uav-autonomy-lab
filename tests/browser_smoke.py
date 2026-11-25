"""Explicit browser integration check: python tests/browser_smoke.py [--live].

Requires Playwright and an installed Chrome. --live exercises the deployed API
and a single synthetic-image L40S inference, using the local ignored access code.
--connect checks the real runtime connection without allocating GPU inference.
"""
import argparse
import json
from pathlib import Path
import re

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def verify_connection_card(browser, url, errors):
    """Exercise authorization removal and changing source metadata with a stub API."""
    page = browser.new_page(viewport={"width": 1440, "height": 950})
    page.on("pageerror", lambda error: errors.append(str(error)))
    runtime = {"configured_gpu": "A100", "configured_gpu_vram_gb": 80, "perception": "on demand", "python": "3.11", "model_count": 1}

    def respond(route):
        if route.request.headers.get("x-lab-key") != "test-connection-key":
            route.fulfill(status=401, json={"detail": "Invalid lab access code"})
        else:
            route.fulfill(json=runtime)

    page.route("**/api/runtime", respond)
    page.route("**/api/edge-deployment", lambda route: route.fulfill(json={"artifacts": [{"format": "PT", "count": 1}, {"format": "ONNX", "count": 1}, {"format": "ENGINE", "count": 0}], "benchmark_target": "jetson-orin-nx", "benchmark": {"p50_ms": 12.5, "p95_ms": 16, "fps": 70}}))
    page.goto(url, wait_until="networkidle")
    expect(page.locator("#connect-button")).to_be_visible()
    expect(page.locator("#compute-source")).to_be_hidden()
    assert page.locator("#connect-button").evaluate("e => {const s=getComputedStyle(e);return s.borderTopWidth==='1px'&&s.borderTopStyle==='solid'}")
    page.locator("#connect-button").click()
    page.locator("#access-code").fill("incorrect-key")
    page.locator("#connect-submit").click()
    expect(page.locator("#connect-error")).to_contain_text("Invalid lab access code")
    expect(page.locator("#compute-source")).to_be_hidden()
    assert page.evaluate("sessionStorage.getItem('uav-lab-access')") is None
    page.locator("#access-code").fill("test-connection-key")
    page.locator("#connect-submit").click()
    expect(page.locator("#compute-gpu")).to_have_text("NVIDIA A100")
    expect(page.locator("#compute-vram")).to_have_text("80 GB · ON DEMAND")
    expect(page.locator("#connect-button")).to_be_hidden()
    assert page.locator("#compute-source").evaluate("e => getComputedStyle(e).borderTopWidth==='0px'")
    page.locator("[data-view=edge]").click()
    expect(page.locator("#edge-artifacts table")).to_be_visible()
    expect(page.locator("#edge-benchmark")).to_contain_text("12.5 ms")
    page.locator("#edge-target").select_option("raspberry-pi-cm5")
    expect(page.locator("#edge-jetson-commands")).to_be_hidden()
    expect(page.locator("#edge-benchmark")).to_contain_text("NO BENCHMARK LOADED")
    assert page.locator("#edge-benchmark .edge-benchmark-metrics").count() == 0
    page.locator("#edge-target").select_option("jetson-orin-nx")
    expect(page.locator("#edge-benchmark")).to_contain_text("12.5 ms")
    page.locator("#header-disconnect-button").click()
    expect(page.locator("#connect-button")).to_be_visible()
    expect(page.locator("#compute-source")).to_be_hidden()
    assert page.evaluate("sessionStorage.getItem('uav-lab-access')") is None
    assert page.locator("#edge-artifacts table").count() == 0
    assert page.locator("#edge-benchmark .edge-benchmark-metrics").count() == 0
    page.reload(wait_until="networkidle")
    expect(page.locator("#connect-button")).to_be_visible()
    expect(page.locator("#compute-source")).to_be_hidden()
    runtime.update(configured_gpu="RTX 3060", configured_gpu_vram_gb=12)
    page.locator("#connect-button").click()
    page.locator("#access-code").fill("test-connection-key")
    page.locator("#connect-submit").click()
    expect(page.locator("#compute-gpu")).to_have_text("NVIDIA RTX 3060")
    expect(page.locator("#compute-vram")).to_have_text("12 GB · ON DEMAND")
    page.reload(wait_until="networkidle")
    expect(page.locator("#compute-source")).to_be_visible()
    expect(page.locator("#compute-vram")).to_have_text("12 GB · ON DEMAND")
    page.set_viewport_size({"width": 390, "height": 844})
    expect(page.locator("#header-disconnect-button")).to_be_visible()
    page.wait_for_function("document.documentElement.scrollWidth <= innerWidth")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.close()
    print("BROWSER_CONNECTION_CARD_PASS: stub runtime; rejected key, source metadata changes, disconnect clears access and edge data, benchmarks stay with their target", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--connect", action="store_true", help="Check deployed connection controls without GPU inference")
    parser.add_argument("--url", default="http://127.0.0.1:8502/uav-sim-lab/")
    args = parser.parse_args()
    output = ROOT / ".browser-test"
    output.mkdir(exist_ok=True)
    errors = []
    with sync_playwright() as runner:
        browser = runner.chromium.launch(executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1150}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="networkidle")
        expect(page.locator("#metric-error")).to_contain_text("0.48")
        expect(page.locator(".intro-subtitle")).to_have_text("SIMULATION ENVIRONMENT FOR GPS DENIED UAV AUTONOMY")
        assert page.locator(".masthead,.intro-copy").count() == 0
        assert page.locator("iframe,video").count() == 0
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert page.evaluate("[...document.querySelectorAll('button,input,select,.panel')].every(e=>parseFloat(getComputedStyle(e).borderTopLeftRadius)===0)")
        assert page.locator("[name=detection_probability]").input_value() == "0.92"
        assert page.locator("[name=sensor_noise]").input_value() == "1"
        page.screenshot(path=str(output / "desktop.png"), full_page=True)
        page.locator("#play-button").click()
        expect(page.locator("#play-button")).to_have_text("PAUSE")
        expect(page.locator("#scene-time")).not_to_have_text("T+60.0 s")
        page.locator("#play-button").click()
        seek = page.locator("#replay-seek")
        seek.focus()
        seek.press("Home")
        seek.press("ArrowRight")
        expect(page.locator("#scene-time")).to_have_text("T+0.2 s")
        page.locator("#mode-3d").click()
        page.wait_for_function("document.getElementById('scene-3d').data?.some(t=>t.type==='scatter3d')")
        page.screenshot(path=str(output / "three-dimensional.png"), full_page=False)
        page.locator("#mode-2d").click()
        for workspace in ["perception", "evaluation", "edge", "deployment", "architecture", "mission"]:
            page.locator(f"[data-view={workspace}]").click()
            expect(page.locator(f"#view-{workspace}")).to_be_visible()
        page.evaluate("location.hash = 'edge'")
        expect(page.locator("#view-edge")).to_be_visible()
        page.evaluate("location.hash = 'mission'")
        expect(page.locator("#view-mission")).to_be_visible()
        with page.expect_download() as downloaded:
            page.locator("#export-json").click()
        downloaded.value.save_as(str(output / "experiment.json"))
        assert len(json.loads((output / "experiment.json").read_text())["samples"]) == 600
        print("BROWSER_STATIC_PASS: replay, synchronized telemetry, 3D, navigation, exports, square controls", flush=True)
        verify_connection_card(browser, args.url, errors)

        if args.live or args.connect:
            code = (ROOT / ".env.modal").read_text().split("=", 1)[1].strip()
            page.locator("#connect-button").click()
            page.locator("#access-code").fill(code)
            page.locator("#connect-submit").click()
            expect(page.locator("#compute-source")).to_be_visible(timeout=120000)
            expect(page.locator("#compute-gpu")).to_have_text("NVIDIA L40S")
            expect(page.locator("#compute-vram")).to_have_text("48 GB · ON DEMAND")
            page.reload(wait_until="networkidle")
            expect(page.locator("#compute-source")).to_be_visible(timeout=120000)
            page.locator("#header-disconnect-button").click()
            expect(page.locator("#connect-button")).to_be_visible()
            assert page.evaluate("sessionStorage.getItem('uav-lab-access')") is None
            page.locator("#connect-button").click()
            page.locator("#access-code").fill(code)
            page.locator("#connect-submit").click()
            expect(page.locator("#compute-source")).to_be_visible(timeout=120000)
            expect(page.locator("#compute-vram")).to_have_text("48 GB · ON DEMAND")
            page.screenshot(path=str(output / "connected.png"), full_page=True)
            page.locator("[data-view=edge]").click()
            expect(page.locator("#edge-artifacts table")).to_be_visible(timeout=120000)
            page.screenshot(path=str(output / "edge-deployment.png"), full_page=True)
            page.locator("[data-view=mission]").click()
            print("BROWSER_REAL_CONNECTION_PASS: authenticated L40S/48 GB metadata, restored session, disconnect and reconnect", flush=True)

        if args.live:
            from PIL import Image
            page.locator("[name=scenario]").select_option("Basic Tracking")
            page.locator("[name=duration]").select_option("30")
            page.locator("[name=seed]").fill("43")
            page.locator("#run-button").click()
            expect(page.locator("#run-source")).to_have_text("COMPUTED EXPERIMENT", timeout=120000)
            expect(page.locator("#run-summary")).to_contain_text("Basic Tracking / Seed 43 / 30 s")
            if page.locator("#play-button").inner_text() == "PAUSE":
                page.locator("#play-button").click()
            page.locator("[data-analysis=comparison]").click()
            page.locator("#compare-button").click()
            expect(page.locator("#comparison-results table")).to_be_visible(timeout=120000)
            page.locator("#reference-button").click()
            assert page.locator("#comparison-results table").count() == 0
            page.locator("#history-button").click()
            expect(page.locator("#history-content .history-entry").first).to_be_visible(timeout=120000)
            page.locator("#history-dialog [data-close]").click()
            page.locator("[data-view=perception]").click()
            page.wait_for_function("document.getElementById('model-select').value.length > 0")
            Image.new("RGB", (640, 640)).save(output / "synthetic.png")
            page.locator("#media-file").set_input_files(str(output / "synthetic.png"))
            page.locator("#perception-button").click()
            expect(page.locator("#perception-metrics")).to_be_visible(timeout=600000)
            expect(page.locator("#perception-metrics")).to_contain_text("L40S")
            page.screenshot(path=str(output / "perception.png"), full_page=True)
            print("BROWSER_LIVE_PASS: authenticated API, simulation, controlled comparison, history, L40S image inference", flush=True)
        mobile = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        mobile.goto(args.url, wait_until="networkidle")
        expect(mobile.locator("#metric-error")).to_contain_text("0.48")
        assert mobile.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Mobile layout overflows horizontally"
        mobile.screenshot(path=str(output / "mobile.png"), full_page=True)
        mobile.locator("[data-view=edge]").click()
        expect(mobile.locator("#view-edge")).to_be_visible()
        mobile.locator("#edge-target").select_option("versal-vc1902")
        expect(mobile.locator("#edge-target-note")).to_contain_text("listed for selection")
        assert mobile.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Edge deployment overflows on mobile"
        mobile.screenshot(path=str(output / "edge-mobile.png"), full_page=True)
        browser.close()
    if errors:
        raise RuntimeError(errors)
    print("BROWSER_PASS: desktop and mobile; no uncaught JavaScript errors", flush=True)


if __name__ == "__main__":
    main()
