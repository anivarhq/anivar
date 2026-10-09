import { describe, it, expect, beforeEach, vi } from "vitest";
import { storeCamSource, writeCamSource, withUrlCredentials, hideUrlLogin } from "./camSource";

const store = new Map<string, string>();
vi.stubGlobal("localStorage", {
  getItem: (k: string) => store.get(k) ?? null,
  setItem: (k: string, v: string) => { store.set(k, v); },
  removeItem: (k: string) => { store.delete(k); },
});

describe("camSource", () => {
  beforeEach(() => store.clear());

  it("never stores a camera's URL or login", () => {
    writeCamSource(2, { source_type: "rtsp", source_url: "rtsp://admin:pw@10.0.0.5/s1", device_id: "", name: "Gate" });
    storeCamSource(3, { kind: "mjpeg", url: "http://10.0.0.6/video", label: "Yard", authUser: "u", authPass: "pw" });
    expect(JSON.parse(localStorage.getItem("cam_source_2")!)).toEqual({ kind: "rtsp", label: "Gate" });
    expect(JSON.parse(localStorage.getItem("cam_source_3")!)).toEqual({ kind: "mjpeg", label: "Yard" });
  });

  it("puts a typed login into the URL once", () => {
    expect(withUrlCredentials("http://10.0.0.6:8080/video", "admin", "p@ss:w/rd"))
      .toBe("http://admin:p%40ss%3Aw%2Frd@10.0.0.6:8080/video");
    expect(withUrlCredentials("rtsp://10.0.0.5/s1", "viewer", "")).toBe("rtsp://viewer@10.0.0.5/s1");
    expect(withUrlCredentials("rtsp://old:pw@10.0.0.5/s1", "admin", "new")).toBe("rtsp://old:pw@10.0.0.5/s1");
    expect(withUrlCredentials("rtsp://10.0.0.5/s1", "", "")).toBe("rtsp://10.0.0.5/s1");
    expect(withUrlCredentials("10.0.0.5", "admin", "pw")).toBe("10.0.0.5");
  });

  it("hides the login in messages", () => {
    expect(hideUrlLogin("rtsp://admin:pw@10.0.0.5/s1")).toBe("rtsp://10.0.0.5/s1");
    expect(hideUrlLogin("http://10.0.0.6/a@b")).toBe("http://10.0.0.6/a@b");
  });
});
