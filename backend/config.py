def _apply_env_overrides(self) -> None:
        self.development = os.getenv("DEVELOPMENT", str(self.development)).lower() == "true"
        self.documents_folder = os.getenv("DOCUMENTS_FOLDER", self.documents_folder)
        
        # CRITICAL FIX: Apply environment override FIRST
        self.environment = os.getenv("ENVIRONMENT", self.environment)
        
        # Apply environment-specific Weaviate URL AFTER environment is set
        if hasattr(self, 'weaviate_environments') and self.environment in self.weaviate_environments:
            self.weaviate.url = self.weaviate_environments[self.environment]["url"]
        
        # Environment variable overrides
        if os.getenv("WEAVIATE_URL"):
            self.weaviate.url = os.getenv("WEAVIATE_URL")
        if os.getenv("ANTHROPIC_API_KEY"):
            self.claude.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # IMPORTANT: Apply Railway-specific environment variables for Neo4j
        if self.environment == "railway":
            # Set Railway Neo4j config from environment variables or use hardcoded values
            if not self.neo4j.railway:
                self.neo4j.railway = Neo4jEnvironmentConfig()
            
            # Use environment variables if available, otherwise use Railway defaults
            self.neo4j.railway.uri = os.getenv("NEO4J_URI", "bolt://turntable.proxy.rlwy.net:43560")
            self.neo4j.railway.user = os.getenv("NEO4J_USER", "neo4j")
            self.neo4j.railway.password = os.getenv("NEO4J_PASSWORD", "mg89wxdi38d1xytau4yd40d7telcxtqo")
            self.neo4j.railway.database = os.getenv("NEO4J_DATABASE", "neo4j")
            
            print(f"🚀 Railway Neo4j configured: {self.neo4j.railway.uri}")
        
        # Handle environment variables for production Neo4j
        if os.getenv("NEO4J_URI"):
            if self.neo4j.production is None:
                self.neo4j.production = Neo4jEnvironmentConfig()
            self.neo4j.production.uri = os.getenv("NEO4J_URI")
        if os.getenv("NEO4J_USER"):
            if self.neo4j.production is None:
                self.neo4j.production = Neo4jEnvironmentConfig()
            self.neo4j.production.user = os.getenv("NEO4J_USER")
        if os.getenv("NEO4J_PASSWORD"):
            if self.neo4j.production is None:
                self.neo4j.production = Neo4jEnvironmentConfig()
            self.neo4j.production.password = os.getenv("NEO4J_PASSWORD")
        if os.getenv("NEO4J_DATABASE"):
            if self.neo4j.production is None:
                self.neo4j.production = Neo4jEnvironmentConfig()
            self.neo4j.production.database = os.getenv("NEO4J_DATABASE")