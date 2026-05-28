"""OPC-UA client for reading sensor data from manufacturing equipment."""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator
from asyncua import Client, Node

logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    timestamp: datetime
    node_id: str
    value: float
    quality: str

class OPCUAIngestionClient:
    """Reads real-time sensor data from OPC-UA servers (Industry 4.0)."""

    def __init__(self, endpoint: str, username: str = None, password: str = None):
        self.endpoint = endpoint
        self.username = username
        self.password = password
        self._client = None

    async def __aenter__(self):
        self._client = Client(url=self.endpoint)
        if self.username:
            self._client.set_user(self.username)
            self._client.set_password(self.password)
        await self._client.connect()
        logger.info(f"Connected to OPC-UA server: {self.endpoint}")
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.disconnect()

    async def stream_readings(
        self, node_ids: list[str], interval_ms: int = 100
    ) -> AsyncGenerator[SensorReading, None]:
        """Stream sensor readings at the specified interval."""
        nodes: list[Node] = [self._client.get_node(nid) for nid in node_ids]
        while True:
            for node, node_id in zip(nodes, node_ids):
                try:
                    value = await node.read_value()
                    yield SensorReading(
                        timestamp=datetime.utcnow(),
                        node_id=node_id,
                        value=float(value),
                        quality="Good",
                    )
                except Exception as e:
                    logger.warning(f"Failed to read {node_id}: {e}")
                    yield SensorReading(
                        timestamp=datetime.utcnow(),
                        node_id=node_id,
                        value=float("nan"),
                        quality="Bad",
                    )
            await asyncio.sleep(interval_ms / 1000)
