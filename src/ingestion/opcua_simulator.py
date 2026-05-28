"""
Simulated OPC-UA server for development and testing.
Emulates a manufacturing line with vibration, temperature, and pressure sensors.
Run this instead of a real PLC when developing locally.
"""
import asyncio
import random
import math
from datetime import datetime
from asyncua import Server
from asyncua import ua

async def run_simulator(endpoint: str = "opc.tcp://0.0.0.0:4840/manufacturing/"):
    server = Server()
    await server.init()
    server.set_endpoint(endpoint)
    server.set_server_name("Manufacturing Line Simulator")

    uri = "http://manufacturing-simulator.local"
    idx = await server.register_namespace(uri)

    objects = server.nodes.objects
    line = await objects.add_object(idx, "ProductionLine01")

    # Sensor nodes
    vibration = await line.add_variable(idx, "Vibration_mm_s", 0.0)
    temperature = await line.add_variable(idx, "Temperature_C", 25.0)
    pressure = await line.add_variable(idx, "Pressure_bar", 4.0)
    cycle_count = await line.add_variable(idx, "CycleCount", 0)
    fault_code = await line.add_variable(idx, "FaultCode", 0)

    await vibration.set_writable()
    await temperature.set_writable()
    await pressure.set_writable()

    print(f"OPC-UA Simulator running at {endpoint}")
    print("Node IDs:")
    print(f"  Vibration:   ns={idx};i={vibration.nodeid.Identifier}")
    print(f"  Temperature: ns={idx};i={temperature.nodeid.Identifier}")
    print(f"  Pressure:    ns={idx};i={pressure.nodeid.Identifier}")

    count = 0
    async with server:
        while True:
            # Normal operating range with realistic noise
            t = count / 10.0
            vib = 2.5 + 0.5 * math.sin(t) + random.gauss(0, 0.1)
            temp = 85.0 + 3.0 * math.sin(t / 5) + random.gauss(0, 0.5)
            pres = 4.1 + 0.2 * math.sin(t / 3) + random.gauss(0, 0.05)

            # Inject anomaly every ~200 cycles (5% rate)
            if count % 200 == 0 and count > 0:
                vib += random.uniform(5, 15)    # vibration spike
                temp += random.uniform(10, 25)  # overheating
                print(f"[{datetime.utcnow().isoformat()}] ANOMALY INJECTED at cycle {count}")

            await vibration.write_value(round(vib, 3))
            await temperature.write_value(round(temp, 2))
            await pressure.write_value(round(pres, 3))
            await cycle_count.write_value(count)

            count += 1
            await asyncio.sleep(0.1)  # 10 Hz

if __name__ == "__main__":
    asyncio.run(run_simulator())
