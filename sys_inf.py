import platform
 
my_system = platform.uname()
 
print(f"System: {my_system.system}")
print(f"Node Name: {my_system.node}")
print(f"Release: {my_system.release}")
print(f"Version: {my_system.version}")
print(f"Machine: {my_system.machine}")
print(f"Processor: {my_system.processor}")

# importing module
import platform
  
# dictionary
info = {}
  
# platform details
platform_details = platform.platform()
  
# adding it to dictionary
info["platform details"] = platform_details
  
# system name
system_name = platform.system()
  
# adding it to dictionary
info["system name"] = system_name
  
# processor name
processor_name = platform.processor()
  
# adding it to dictionary
info["processor name"] = processor_name
  
# architectural detail
architecture_details = platform.architecture()
  
# adding it to dictionary
info["architectural detail"] = architecture_details
  
# printing the details
for i, j in info.items():
    print(i, " - ", j)

import platform

#Computer network name
print(f"Computer network name: {platform.node()}")
#Machine type
print(f"Machine type: {platform.machine()}")
#Processor type
print(f"Processor type: {platform.processor()}")
#Platform type
print(f"Platform type: {platform.platform()}")
#Operating system
print(f"Operating system: {platform.system()}")
#Operating system release
print(f"Operating system release: {platform.release()}")
#Operating system version
print(f"Operating system version: {platform.version()}")
