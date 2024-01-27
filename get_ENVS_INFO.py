
import os
import pkg_resources
dists = [d for d in pkg_resources.working_set]
# Save environment information
with open('environment.txt', 'w') as f:
    f.write(str(os.environ))

# Save package information
with open('requirements.txt', 'w') as f:
    for package in pip.get_installed_distributions():
        f.write(package.project_name + '==' + package.version + '\n')