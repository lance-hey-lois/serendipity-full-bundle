#!/usr/bin/env python3
"""
Simple Serendipity Engine UI Startup
===================================

A streamlined startup script that bypasses MongoDB SSL issues
and starts the essential services for demonstration.
"""

import os
import sys
import subprocess
import time
import signal
import threading
from pathlib import Path

class SimpleStartup:
    def __init__(self):
        self.processes = {}
        self.shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        print(f"\n🛑 Received shutdown signal {signum}")
        self.shutdown_requested = True
        self.stop_all_services()
        sys.exit(0)
    
    def start_backend_service(self, name: str, script: str, port: int, env_vars: dict = None):
        """Start a backend service"""
        if not Path(script).exists():
            print(f"❌ Script not found: {script}")
            return False
        
        print(f"🚀 Starting {name} on port {port}...")
        
        # Create environment
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        
        # Add MongoDB SSL bypass for testing
        env['MONGODB_URI'] = env.get('MONGODB_URI', '').replace('?retryWrites=true&w=majority', 
                                                              '?retryWrites=true&w=majority&ssl=false&ssl_cert_reqs=CERT_NONE')
        
        try:
            # Start the process
            process = subprocess.Popen(
                [sys.executable, script],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes[name] = process
            
            # Monitor output in background
            def monitor_output():
                try:
                    for line in process.stdout:
                        if self.shutdown_requested:
                            break
                        if line.strip():
                            print(f"[{name}] {line.strip()}")
                except:
                    pass
            
            thread = threading.Thread(target=monitor_output)
            thread.daemon = True
            thread.start()
            
            # Wait briefly to see if it starts successfully
            time.sleep(3)
            
            if process.poll() is None:
                print(f"✅ {name} started successfully (PID: {process.pid})")
                return True
            else:
                print(f"❌ {name} failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return False
    
    def start_frontend_service(self):
        """Start React frontend"""
        react_dir = Path("quantum-discovery-react")
        
        if not react_dir.exists():
            print("❌ React directory not found")
            return False
        
        print("🚀 Starting React frontend...")
        
        try:
            env = os.environ.copy()
            env['BROWSER'] = 'none'  # Don't auto-open browser
            
            process = subprocess.Popen(
                ['npm', 'start'],
                cwd=react_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes['frontend'] = process
            
            # Monitor output
            def monitor_frontend():
                try:
                    for line in process.stdout:
                        if self.shutdown_requested:
                            break
                        if line.strip():
                            print(f"[Frontend] {line.strip()}")
                            
                            # Check for successful compilation
                            if "webpack compiled" in line or "Compiled successfully" in line:
                                print("✅ React frontend compiled successfully!")
                except:
                    pass
            
            thread = threading.Thread(target=monitor_frontend)
            thread.daemon = True
            thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
    
    def stop_all_services(self):
        """Stop all running services"""
        print("\n🛑 Stopping all services...")
        
        for name, process in self.processes.items():
            try:
                if process.poll() is None:  # Still running
                    print(f"   Stopping {name}...")
                    process.terminate()
                    
                    # Give it time to stop gracefully
                    try:
                        process.wait(timeout=5)
                        print(f"✅ {name} stopped gracefully")
                    except subprocess.TimeoutExpired:
                        process.kill()
                        print(f"⚠️  {name} force killed")
                        
            except Exception as e:
                print(f"⚠️  Error stopping {name}: {e}")
        
        print("✅ All services stopped")
    
    def check_services(self):
        """Check which services are running"""
        import requests
        
        services = [
            ('Backend Main', 'http://localhost:8000/'),
            ('Serendipity API', 'http://localhost:8078/'),
            ('Quantum API', 'http://localhost:8080/'),
            ('Frontend', 'http://localhost:3000/')
        ]
        
        print("\n📊 Service Status Check:")
        print("-" * 40)
        
        for name, url in services:
            try:
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    print(f"✅ {name} - Running")
                else:
                    print(f"⚠️  {name} - Status {response.status_code}")
            except requests.exceptions.RequestException:
                print(f"❌ {name} - Not responding")
    
    def run(self):
        """Run the complete startup sequence"""
        print("🚀 SERENDIPITY ENGINE UI - SIMPLE STARTUP")
        print("=" * 50)
        
        # Start backend services
        backend_services = [
            ('Backend Main', 'backend/main.py', 8000),
            ('Serendipity API', 'serendipity_api.py', 8078)
        ]
        
        started_services = []
        
        for name, script, port in backend_services:
            if self.start_backend_service(name, script, port):
                started_services.append(name)
                time.sleep(2)  # Stagger startup
        
        if not started_services:
            print("❌ No backend services started. Exiting.")
            return False
        
        # Start frontend
        frontend_started = self.start_frontend_service()
        
        # Wait for services to fully initialize
        print("\n⏳ Waiting for services to initialize...")
        time.sleep(10)
        
        # Check service status
        self.check_services()
        
        print("\n✅ SERENDIPITY ENGINE UI - READY!")
        print("=" * 50)
        print("🌐 Access Points:")
        print("   Frontend:      http://localhost:3000")
        print("   Main API:      http://localhost:8000/docs")  
        print("   Serendipity:   http://localhost:8078")
        print("\nPress Ctrl+C to stop all services")
        
        # Keep running until interrupted
        try:
            while not self.shutdown_requested:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_all_services()
        
        return True

def main():
    """Main entry point"""
    startup = SimpleStartup()
    
    try:
        return startup.run()
    except KeyboardInterrupt:
        print("\n🛑 Startup interrupted")
        startup.stop_all_services()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)