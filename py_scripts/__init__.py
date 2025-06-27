from .app.webTools import Navigate2Page

from .LLM_module.training.training import train

__version__ = "0.1.0"
__package__ = "aitrainmix"
__author__ = "Tourist Chen"
# __import__("pkg_resources").declare_namespace(__name__)
__doc__ = "This is a package for AI Training Running Mix"
__path__ = __import__('pkgutil').extend_path(__path__, __name__)


__all__ = [
    'Navigate2Page',
    'train'
    ]