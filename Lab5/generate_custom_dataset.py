import struct
import random
import math
import sys
import os

def generate_stable_system(bodies_per_system):
    bodies = []
    
    # 1. Massive black hole in the center
    center_mass = 1000.0
    bodies.append({
        'x': 0.0, 'y': 0.0, 'z': 0.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0
    })
    
    # 2. Orbiting stars
    for _ in range(bodies_per_system - 1):
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(50.0, 500.0)
        # v = sqrt(GM/r), simplified G=1
        velocity = math.sqrt(center_mass / dist)
        
        # Position
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        z = random.uniform(-10, 10)
        
        # Velocity (Tangential)
        vx = -velocity * math.sin(angle)
        vy = velocity * math.cos(angle)
        vz = random.uniform(-0.1, 0.1)
        
        bodies.append({
            'x': x, 'y': y, 'z': z,
            'vx': vx, 'vy': vy, 'vz': vz
        })

    return bodies

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 generate_custom_dataset.py <num_systems> <bodies_per_system> [output_filename]")
        return

    num_systems = int(sys.argv[1])
    bodies_per_system = int(sys.argv[2])
    
    if len(sys.argv) > 3:
        filename = sys.argv[3]
    else:
        filename = f"galaxy_{num_systems}x{bodies_per_system}.bin"

    # Ensure Inputs directory exists
    os.makedirs("Inputs", exist_ok=True)
    filepath = os.path.join("Inputs", filename)

    with open(filepath, 'wb') as f:
        # Write Header: Num Systems, Bodies per System
        f.write(struct.pack('ii', num_systems, bodies_per_system))
        
        for i in range(num_systems):
            system_data = generate_stable_system(bodies_per_system)
            for b in system_data:
                # Write x, y, z, vx, vy, vz (floats)
                data = struct.pack('ffffff', 
                                   b['x'], b['y'], b['z'], 
                                   b['vx'], b['vy'], b['vz'])
                f.write(data)

    print(f"Generated {filepath} with {num_systems} systems of {bodies_per_system} bodies.")

if __name__ == "__main__":
    main()
