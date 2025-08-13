#!/usr/bin/env python3
"""
Serendipity Engine UI - Agent Swarm Startup Coordinator
======================================================

Initializes and coordinates a specialized agent swarm to handle:
1. Database verification agent
2. Backend service startup agent  
3. Frontend service startup agent
4. Integration testing agent
5. System monitoring agent
"""

import asyncio
import subprocess
import time
import json
import signal
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

@dataclass
class AgentStatus:
    name: str
    status: str  # 'initializing', 'running', 'completed', 'failed'
    pid: Optional[int] = None
    port: Optional[int] = None
    log_file: Optional[str] = None
    start_time: Optional[float] = None
    last_health_check: Optional[float] = None

class SwarmCoordinator:
    def __init__(self):
        self.agents: Dict[str, AgentStatus] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.shutdown_requested = False
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Service ports
        self.ports = {
            'backend_main': 8000,
            'serendipity_api': 8078,
            'quantum_api': 8080,
            'frontend': 3000
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        print(f"\n🛑 Received shutdown signal {signum}")
        self.shutdown_requested = True
        self.shutdown_all_agents()
    
    def initialize_agent(self, agent_name: str, **kwargs) -> AgentStatus:
        """Initialize a new agent"""
        agent = AgentStatus(
            name=agent_name,
            status='initializing',
            start_time=time.time(),
            **kwargs
        )
        self.agents[agent_name] = agent
        print(f"🤖 Initializing agent: {agent_name}")
        return agent
    
    def update_agent_status(self, agent_name: str, status: str, **kwargs):
        """Update agent status"""
        if agent_name in self.agents:
            agent = self.agents[agent_name]
            agent.status = status
            for key, value in kwargs.items():
                setattr(agent, key, value)
            print(f"📊 Agent {agent_name}: {status}")
    
    async def database_verification_agent(self) -> bool:
        """Agent specialized in database connectivity verification"""
        agent = self.initialize_agent('database_verifier')
        
        try:
            self.update_agent_status('database_verifier', 'running')
            
            # Import MongoDB components
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
            from dotenv import load_dotenv
            
            # Load environment
            load_dotenv('.env')
            
            mongodb_uri = os.getenv("MONGODB_URI")
            db_name = os.getenv("DB_NAME", "MagicCRM")
            
            if not mongodb_uri:
                raise ValueError("MONGODB_URI not found in environment")
            
            # Test connection with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"🔍 Database Agent: Testing MongoDB connection (attempt {attempt + 1}/{max_retries})")
                    
                    client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=10000)
                    
                    # Ping the server
                    client.admin.command('ping')
                    
                    # Test database access
                    db = client[db_name]
                    collections = db.list_collection_names()
                    
                    # Test key collection
                    profiles_count = 0
                    if 'public_profiles' in collections:
                        profiles_count = db['public_profiles'].count_documents({})
                    
                    print(f"✅ Database Agent: Connected to MongoDB")
                    print(f"   Database: {db_name}")
                    print(f"   Collections: {len(collections)}")
                    print(f"   Profiles: {profiles_count}")
                    
                    client.close()
                    self.update_agent_status('database_verifier', 'completed')
                    return True
                    
                except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"⚠️  Database Agent: Connection attempt failed, retrying...")
                    await asyncio.sleep(2)
            
        except Exception as e:
            print(f"❌ Database Agent: Failed - {e}")
            self.update_agent_status('database_verifier', 'failed')
            return False
    
    async def backend_service_agent(self, service_name: str, script_path: str, port: int) -> bool:
        """Agent specialized in backend service startup and health monitoring"""
        agent_name = f'backend_{service_name}'
        agent = self.initialize_agent(agent_name, port=port)
        
        try:
            self.update_agent_status(agent_name, 'running')
            
            # Create log file
            log_file = self.logs_dir / f"{service_name}.log"
            agent.log_file = str(log_file)
            
            print(f"🚀 Backend Agent ({service_name}): Starting service on port {port}")
            
            # Start the service
            cmd = ['python3', script_path]
            
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid  # Create new process group
                )
            
            self.processes[agent_name] = process
            agent.pid = process.pid
            
            # Wait for service to start
            print(f"🔍 Backend Agent ({service_name}): Waiting for service startup...")
            
            # Health check with timeout
            import requests
            max_wait = 30  # seconds
            start_wait = time.time()
            
            while time.time() - start_wait < max_wait:
                if self.shutdown_requested:
                    return False
                
                try:
                    response = requests.get(f"http://localhost:{port}/", timeout=5)
                    if response.status_code == 200:
                        print(f"✅ Backend Agent ({service_name}): Service healthy on port {port}")
                        self.update_agent_status(agent_name, 'completed')
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                await asyncio.sleep(1)
            
            print(f"❌ Backend Agent ({service_name}): Service failed to start within {max_wait}s")
            self.update_agent_status(agent_name, 'failed')
            return False
            
        except Exception as e:
            print(f"❌ Backend Agent ({service_name}): Failed - {e}")
            self.update_agent_status(agent_name, 'failed')
            return False
    
    async def frontend_service_agent(self) -> bool:
        """Agent specialized in React frontend startup"""
        agent_name = 'frontend_service'
        agent = self.initialize_agent(agent_name, port=3000)
        
        try:
            self.update_agent_status(agent_name, 'running')
            
            react_dir = Path("quantum-discovery-react")
            if not react_dir.exists():
                raise ValueError("React directory not found")
            
            # Create log file
            log_file = self.logs_dir / "frontend.log"
            agent.log_file = str(log_file)
            
            print("🚀 Frontend Agent: Starting React development server")
            
            # Start React development server
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    ['npm', 'start'],
                    cwd=react_dir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                    env={**os.environ, 'BROWSER': 'none'}  # Don't auto-open browser
                )
            
            self.processes[agent_name] = process
            agent.pid = process.pid
            
            # Wait for React to compile and start
            print("🔍 Frontend Agent: Waiting for React compilation...")
            
            import requests
            max_wait = 60  # React can take longer to compile
            start_wait = time.time()
            
            while time.time() - start_wait < max_wait:
                if self.shutdown_requested:
                    return False
                
                try:
                    response = requests.get("http://localhost:3000/", timeout=5)
                    if response.status_code == 200:
                        print("✅ Frontend Agent: React app running on port 3000")
                        self.update_agent_status(agent_name, 'completed')
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                await asyncio.sleep(2)
            
            print(f"❌ Frontend Agent: React failed to start within {max_wait}s")
            self.update_agent_status(agent_name, 'failed')
            return False
            
        except Exception as e:
            print(f"❌ Frontend Agent: Failed - {e}")
            self.update_agent_status(agent_name, 'failed')
            return False
    
    async def integration_testing_agent(self) -> bool:
        """Agent specialized in end-to-end integration testing"""
        agent_name = 'integration_tester'
        agent = self.initialize_agent(agent_name)
        
        try:
            self.update_agent_status(agent_name, 'running')
            print("🧪 Integration Agent: Testing end-to-end connectivity")
            
            import requests
            
            # Test backend endpoints
            backend_tests = [
                ('main', 8000, '/'),
                ('serendipity', 8078, '/api/serendipity/stats'),
                ('quantum', 8080, '/')
            ]
            
            for service, port, endpoint in backend_tests:
                try:
                    url = f"http://localhost:{port}{endpoint}"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        print(f"✅ Integration Agent: {service} service responsive")
                    else:
                        print(f"⚠️  Integration Agent: {service} service returned {response.status_code}")
                        
                except requests.exceptions.RequestException as e:
                    print(f"❌ Integration Agent: {service} service unreachable - {e}")
            
            # Test frontend
            try:
                response = requests.get("http://localhost:3000/", timeout=10)
                if response.status_code == 200:
                    print("✅ Integration Agent: Frontend responsive")
                else:
                    print(f"⚠️  Integration Agent: Frontend returned {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"❌ Integration Agent: Frontend unreachable - {e}")
            
            # Test API integration
            try:
                # Test serendipity API
                response = requests.post(
                    "http://localhost:8078/api/serendipity/discover",
                    json={
                        "query": "test user",
                        "userId": "test",
                        "limit": 5
                    },
                    timeout=15
                )
                
                if response.status_code == 200:
                    print("✅ Integration Agent: Serendipity API integration working")
                else:
                    print(f"⚠️  Integration Agent: Serendipity API returned {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Integration Agent: API integration failed - {e}")
            
            self.update_agent_status(agent_name, 'completed')
            return True
            
        except Exception as e:
            print(f"❌ Integration Agent: Failed - {e}")
            self.update_agent_status(agent_name, 'failed')
            return False
    
    async def system_monitor_agent(self):
        """Agent specialized in continuous system monitoring"""
        agent_name = 'system_monitor'
        agent = self.initialize_agent(agent_name)
        
        try:
            self.update_agent_status(agent_name, 'running')
            print("📊 Monitor Agent: Starting system monitoring")
            
            import requests
            
            while not self.shutdown_requested:
                # Check all services
                for service_name, port in self.ports.items():
                    try:
                        response = requests.get(f"http://localhost:{port}/", timeout=3)
                        status = "healthy" if response.status_code == 200 else f"status_{response.status_code}"
                    except:
                        status = "unreachable"
                    
                    if agent_name in self.agents:
                        self.agents[service_name] = self.agents.get(service_name, AgentStatus(service_name, status))
                        self.agents[service_name].last_health_check = time.time()
                
                await asyncio.sleep(30)  # Check every 30 seconds
            
            print("📊 Monitor Agent: Monitoring stopped")
            self.update_agent_status(agent_name, 'completed')
            
        except Exception as e:
            print(f"❌ Monitor Agent: Failed - {e}")
            self.update_agent_status(agent_name, 'failed')
    
    def shutdown_all_agents(self):
        """Gracefully shutdown all running processes"""
        print("🛑 Shutting down all agents and services...")
        
        for name, process in self.processes.items():
            try:
                print(f"   Stopping {name} (PID: {process.pid})")
                
                # Send SIGTERM to process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    
            except (ProcessLookupError, PermissionError):
                pass  # Process already dead
        
        print("✅ All agents shut down")
    
    async def run_swarm(self):
        """Run the complete agent swarm orchestration"""
        print("🚀 SERENDIPITY ENGINE UI - AGENT SWARM STARTUP")
        print("=" * 60)
        
        try:
            # Phase 1: Database Verification
            print("\n🔍 PHASE 1: Database Verification")
            db_ok = await self.database_verification_agent()
            
            if not db_ok:
                print("❌ Database verification failed. Cannot proceed.")
                return False
            
            # Phase 2: Backend Services
            print("\n🚀 PHASE 2: Backend Services Startup")
            
            backend_services = [
                ('main', 'backend/main.py', self.ports['backend_main']),
                ('serendipity', 'serendipity_api.py', self.ports['serendipity_api']),
                ('quantum', 'quantum_api.py', self.ports['quantum_api'])
            ]
            
            backend_tasks = []
            for service_name, script_path, port in backend_services:
                if Path(script_path).exists():
                    task = self.backend_service_agent(service_name, script_path, port)
                    backend_tasks.append(task)
            
            backend_results = await asyncio.gather(*backend_tasks, return_exceptions=True)
            
            # Check if at least one backend service started
            backend_success = any(result is True for result in backend_results if not isinstance(result, Exception))
            
            if not backend_success:
                print("❌ No backend services started successfully")
                return False
            
            # Phase 3: Frontend Service
            print("\n🎨 PHASE 3: Frontend Service Startup")
            frontend_ok = await self.frontend_service_agent()
            
            # Phase 4: Integration Testing
            print("\n🧪 PHASE 4: Integration Testing")
            await asyncio.sleep(5)  # Let services fully initialize
            await self.integration_testing_agent()
            
            # Phase 5: System Monitoring
            print("\n📊 PHASE 5: System Monitoring")
            monitor_task = asyncio.create_task(self.system_monitor_agent())
            
            # System is operational
            print("\n✅ SERENDIPITY ENGINE UI - SYSTEM OPERATIONAL")
            print("=" * 60)
            self.print_system_status()
            print("\nPress Ctrl+C to shutdown all services")
            
            # Keep running until shutdown
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            
            return True
            
        except Exception as e:
            print(f"❌ Swarm coordination failed: {e}")
            return False
        
        finally:
            if not self.shutdown_requested:
                self.shutdown_all_agents()
    
    def print_system_status(self):
        """Print current system status"""
        print("\n📊 SYSTEM STATUS:")
        print("-" * 40)
        
        for name, agent in self.agents.items():
            status_symbol = {
                'completed': '✅',
                'running': '🔄', 
                'failed': '❌',
                'initializing': '🔄'
            }.get(agent.status, '❓')
            
            port_info = f" (port {agent.port})" if agent.port else ""
            print(f"{status_symbol} {name.replace('_', ' ').title()}: {agent.status}{port_info}")
        
        print("\n🌐 QUICK ACCESS:")
        print(f"   Frontend: http://localhost:3000")
        print(f"   Main API: http://localhost:8000/docs")
        print(f"   Serendipity: http://localhost:8078")
        print(f"   Quantum API: http://localhost:8080")

async def main():
    """Main entry point for agent swarm startup"""
    coordinator = SwarmCoordinator()
    
    try:
        success = await coordinator.run_swarm()
        return success
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
        coordinator.shutdown_all_agents()
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        exit_code = 130  # Standard exit code for Ctrl+C
    
    exit(exit_code)