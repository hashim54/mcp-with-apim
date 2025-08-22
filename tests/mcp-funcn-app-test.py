import json, threading, requests, uuid, time
from urllib.parse import urljoin

class MCPClientError(Exception): ...

class MCPClient:
    def __init__(self, base_sse_url: str, function_key: str | None = None, timeout=30):
        # base_sse_url: https://mcp-test-app1923.azurewebsites.net/runtime/webhooks/mcp/sse?code=4g5ubIpBtocZnIWai5X5C0puyU809W5f4OWUKVwG1-EPAzFug0AGOg==
        self.base_sse_url = base_sse_url
        self.function_key = function_key
        self.timeout = timeout
        self.write_url = None
        self._session = requests.Session()
        self._pending: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._listener_thread = None
        self._stop = threading.Event()

    def connect(self, handshake_timeout=20):
        def _listen():
            with self._session.get(self.base_sse_url, headers={
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }, stream=True, timeout=self.timeout) as r:
                r.raise_for_status()
                event, buf = None, []
                for raw in r.iter_lines(decode_unicode=True):
                    if self._stop.is_set():
                        break
                    if raw is None:
                        continue
                    line = raw.strip()
                    if line == "":
                        if buf:
                            data = "\n".join(buf)
                            self._handle_sse_event(event, data)
                        event, buf = None, []
                        continue
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        buf.append(line[5:].strip())
        self._listener_thread = threading.Thread(target=_listen, daemon=True)
        self._listener_thread.start()

        # Wait for write_url discovery
        start = time.time()
        while self.write_url is None and (time.time() - start) < handshake_timeout:
            time.sleep(0.05)
        if self.write_url is None:
            raise TimeoutError("Did not receive session write endpoint via SSE handshake")

    def _handle_sse_event(self, event: str | None, data: str):
        # Discover endpoint
        if event == "endpoint" and data and self.write_url is None:
            self.write_url = self._qualify_endpoint(data)
            return
        # JSON objects
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return
        if self.write_url is None and isinstance(obj, dict) and "endpoint" in obj:
            self.write_url = self._qualify_endpoint(obj["endpoint"])
            return
        # JSON-RPC responses
        if isinstance(obj, dict) and "id" in obj:
            with self._lock:
                waiter = self._pending.pop(obj["id"], None)
            if waiter is not None:
                waiter["response"] = obj
                waiter["event"].set()

    def _qualify_endpoint(self, endpoint: str) -> str:
        base = self.base_sse_url.split("/sse")[0]
        # endpoint may already include ?code=, ensure function key appended if needed
        if self.function_key and "code=" not in endpoint:
            sep = "&" if "?" in endpoint else "?"
            endpoint = f"{endpoint}{sep}code={self.function_key}"
        return urljoin(base + "/", endpoint)

    def _rpc(self, method: str, params: dict | None) -> dict:
        if self.write_url is None:
            raise MCPClientError("Not connected or write endpoint not discovered")
        rid = str(uuid.uuid4())
        evt = threading.Event()
        holder = {"event": evt}
        with self._lock:
            self._pending[rid] = holder
        payload = {
            "jsonrpc": "2.0",
            "id": rid,
            "method": method,
            "params": params or {}
        }
        resp = self._session.post(self.write_url, json=payload, timeout=self.timeout)
        if resp.status_code >= 400:
            raise MCPClientError(f"HTTP {resp.status_code}: {resp.text}")
        if not evt.wait(self.timeout):
            with self._lock:
                self._pending.pop(rid, None)
            raise MCPClientError(f"Timeout waiting for response to {method}")
        result = holder["response"]
        if "error" in result:
            raise MCPClientError(result["error"])
        return result

    def initialize(self, protocol_version="2024-05-01", client_name="custom-mcp", client_version="0.1"):
        return self._rpc("initialize", {
            "protocolVersion": protocol_version,
            "capabilities": {},
            "client": {"name": client_name, "version": client_version}
        })

    def list_tools(self):
        resp = self._rpc("tools/list", {})
        return resp.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict):
        resp = self._rpc("tools/call", {"name": name, "arguments": arguments})
        return resp.get("result", {}).get("content", [])

    # Convenience wrappers
    def search(self, query: str):
        content = self.call_tool("search", {"query": query})
        # Expect content list with possibly a json block
        for block in content:
            if block.get("type") == "json":
                return block.get("data")
        return content

    def search_by_doc_id(self, doc_id: str):
        content = self.call_tool("search_by_doc_id", {"doc_id": doc_id})
        for block in content:
            if block.get("type") == "json":
                return block.get("data")
        return content

    def close(self):
        self._stop.set()
        if self._listener_thread:
            self._listener_thread.join(timeout=1)

if __name__ == "__main__":
    FUNCTION_KEY = "4g5ubIpBtocZnIWai5X5C0puyU809W5f4OWUKVwG1-EPAzFug0AGOg=="
    sse_url = f"https://mcp-test-app1923.azurewebsites.net/runtime/webhooks/mcp/sse?code={FUNCTION_KEY}"
    client = MCPClient(sse_url, function_key=FUNCTION_KEY)
    client.connect()
    client.initialize()
    print("Client Initialized with default values")
    print("Tools:", client.list_tools())
    docs = client.search("Azure Service Bus Architectures")
    print("Search docs:", json.dumps(docs, indent=2)[:600])
    client.close()