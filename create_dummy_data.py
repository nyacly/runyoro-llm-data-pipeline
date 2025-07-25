import os, pathlib
dummy_dir = pathlib.Path("/tmp/dummy_text")
dummy_dir.mkdir(parents=True, exist_ok=True)
with open(dummy_dir / "sample.txt", "w") as f:
    for i in range(10):
        f.write(f"Dummy sentence {i}\n")
print("Dummy data created at", dummy_dir)


