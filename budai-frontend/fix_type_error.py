import re

with open("app/(protected)/home/DashboardClient.tsx", "r") as f:
    content = f.read()

content = content.replace('variant="bordered"', 'variant="outline"')

with open("app/(protected)/home/DashboardClient.tsx", "w") as f:
    f.write(content)
