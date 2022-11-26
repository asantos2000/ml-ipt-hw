import subprocess
import sys
import pathlib
import pkg_resources
import re


class PackageInstaller:
    def __init__(self, filename='requirement.txt', dry_run=False, interactive_mode=True):
        self.filename = filename
        self.dry_run = dry_run
        self.interactive_mode = interactive_mode

    def install(self, package_list=[]):
        if package_list:
            install_requires = package_list
        else:
            install_requires = pinst.load_package_list_file()

        # Check and return list of packages to be installed
        packages_to_install = pinst.get_packages_to_install(install_requires)
        # Install each package
        for package in packages_to_install:
            pinst.install_package(package)

        print("🏆 Installation completed")

    def load_package_list_file(self, filename=None):
        if not filename:
            filename = self.filename

        with pathlib.Path(filename).open() as requirements_txt:
            package_list = [
                str(requirement)
                for requirement
                in pkg_resources.parse_requirements(requirements_txt)
            ]
        return package_list

    def install_package(self, package):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package])

    def get_packages_to_install(self, package_list):
        installed_packages = pkg_resources.working_set
        installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
                                          for i in installed_packages])
        packages_to_install = list(
            set(package_list) - set(installed_packages_list))

        if self.interactive_mode:
            # Packages that need to be installed
            print('☝️ This program needs to install the packages below:')
            for package in packages_to_install:
                print(f'- {package}')

            # Packages installed with different version

            packages_to_install_name_version = [re.split(
                r'==|>=|<=|<>', package, maxsplit=2) for package in packages_to_install]
            packages_installed_name_version = [re.split(
                r'==|>=|<=|<>', package, maxsplit=2) for package in installed_packages_list]

            packages_to_install_names = [i[0]
                                         for i in packages_to_install_name_version]
            packages_to_install_versions = [i[1] if len(
                                            i) == 2 else "newest" for i in packages_to_install_name_version]
            packages_installed_names = [i[0]
                                        for i in packages_installed_name_version]
            packages_installed_versions = [i[1] if len(
                i) == 2 else "" for i in packages_installed_name_version]

            diff_packages_version = []

            for name in packages_to_install_names:
                try:
                    idx_installed = packages_installed_names.index(name)
                except:
                    continue
                idx_to_install = packages_to_install_names.index(name)
                if packages_installed_versions[idx_installed] != packages_to_install_versions[idx_to_install]:
                    diff_packages_version.append(
                        [name, packages_installed_versions[idx_installed], packages_to_install_versions[idx_to_install]])

            print(
                "⚠️ Those packages are already installed, and could be upgraded or downgraded:")
            if diff_packages_version:
                for package in diff_packages_version:
                    print(
                        f'- {package[0]}: actual version: {package[1]}, try: {package[2]}')
            else:
                print("No differences found.")

            user_input = input('> Do you want to proceed? (y/n): ')
            if user_input.lower() == 'y':
                return packages_to_install
            else:
                packages_to_install = []

        return packages_to_install


if __name__ == '__main__':
    pinst = PackageInstaller(
        filename='requirements.txt', interactive_mode=True)
    
    # Installing using requirements.txt
    #pinst.install()

    # Install from a list
    pinst.install(['pandas', 'plotly>=5.11.0'])

