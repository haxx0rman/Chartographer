"""
BookWorm: Advanced Document/Knowledge Ingestion System

Refactored modular architecture with focused responsibilities:
- models: Data models and schemas
- processors: Document processing and description generation
- knowledge: Knowledge graph management
- generators: Content generation (mindmaps, etc.)
- library: Document library management
- utils: Configuration and utilities
"""

__version__ = "0.1.0"
__author__ = "BookWorm Team"
__description__ = "Advanced document/knowledge ingestion system with LightRAG and mindmap generation"

from .mindmap_generator import MindmapGenerator
from .utils import ChartographerConfig, load_config, setup_logging

# Maintain backward compatibility - these are the main classes users need
__all__ = [
    
    # Generation components
    'MindmapGenerator',
    
    # Configuration
    'ChartographerConfig',
    'load_config',
    'setup_logging'
]
