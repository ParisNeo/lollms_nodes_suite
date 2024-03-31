import importlib
class PackageManager:
    @staticmethod
    def install_package(package_name):
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
        
    @staticmethod
    def check_package_installed(package_name):
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False
        
    @staticmethod
    def safe_import(module_name, library_name=None):
        if not PackageManager.check_package_installed(module_name):
            print(f"{module_name} module not found. Installing...")
            if library_name:
                PackageManager.install_package(library_name)
            else:
                PackageManager.install_package(module_name)
        globals()[module_name] = importlib.import_module(module_name)
        print(f"{module_name} module imported successfully.")