#!/usr/bin/env python3
"""
Build standalone executable for DeepWiki CLI using PyInstaller
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def install_pyinstaller():
    """Install PyInstaller if not available"""
    try:
        import PyInstaller

        print("✅ PyInstaller is already installed")
    except ImportError:
        print("📦 Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def build_executable():
    """Build the standalone executable"""

    # Get the current directory
    current_dir = Path(__file__).parent

    # Define paths
    query_script = current_dir / "query.py"
    configs_dir = current_dir / "configs"
    dist_dir = current_dir / "dist"
    build_dir = current_dir / "build"

    # Clean previous builds
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    if build_dir.exists():
        shutil.rmtree(build_dir)

    print("🔨 Building standalone executable...")

    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",  # Create a single executable file
        "--name",
        "deepwiki",  # Name of the executable
        "--add-data",
        f"{configs_dir}:configs",  # Include config files
        "--hidden-import",
        "adalflow",
        "--hidden-import",
        "omegaconf",
        "--hidden-import",
        "hydra",
        "--hidden-import",
        "faiss",
        "--console",  # Console application
        str(query_script),
    ]

    try:
        subprocess.check_call(cmd)
        print("✅ Executable built successfully!")

        # Find the executable
        if sys.platform == "win32":
            exe_path = dist_dir / "deepwiki.exe"
        else:
            exe_path = dist_dir / "deepwiki"

        if exe_path.exists():
            print(f"📁 Executable location: {exe_path}")
            print(f"📏 File size: {exe_path.stat().st_size / (1024*1024):.1f} MB")

            # Test the executable
            print("\n🧪 Testing executable...")
            try:
                result = subprocess.run(
                    [str(exe_path), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    print("✅ Executable test passed!")
                else:
                    print("⚠️  Executable test failed:")
                    print(result.stderr)
            except subprocess.TimeoutExpired:
                print("⚠️  Executable test timed out")
            except Exception as e:
                print(f"⚠️  Could not test executable: {e}")
        else:
            print("❌ Executable not found!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False

    return True


def main():
    print("🚀 DeepWiki CLI Executable Builder")
    print("=====================================")

    # Install PyInstaller if needed
    install_pyinstaller()

    # Build executable
    success = build_executable()

    if success:
        print("\n✅ Build completed successfully!")
        print("\n📖 Usage:")
        print("   ./dist/deepwiki repo=/path/to/repo question='What does this do?'")
        print("   ./dist/deepwiki --repo /path/to/repo --question 'What does this do?'")
    else:
        print("\n❌ Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
