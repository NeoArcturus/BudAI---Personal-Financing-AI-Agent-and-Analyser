import re

with open("app/(protected)/home/DashboardClient.tsx", "r") as f:
    content = f.read()

# Fix Bell button
old_bell = """            <Badge.Anchor>
              <Button
                isIconOnly
                variant="primary"
                onPress={() => setIsAuditLogOpen(true)}
                className="w-10 h-10 min-w-10 rounded-full"
              >
                <Bell size={16} />
              </Button>
              <Badge color="danger" placement="top-right" size="sm" variant="primary" />
            </Badge.Anchor>"""

new_bell = """            <Badge.Anchor>
              <Button
                isIconOnly
                variant="bordered"
                onPress={() => setIsAuditLogOpen(true)}
                className="w-10 h-10 min-w-10 rounded-full bg-white/5 border-white/10 hover:bg-white/10 text-foreground/70 hover:text-foreground transition-all shadow-inner"
              >
                <Bell size={16} />
              </Button>
              <Badge color="danger" placement="top-right" size="sm" shape="circle" />
            </Badge.Anchor>"""
content = content.replace(old_bell, new_bell)

# Fix Add New Widget button
old_add_widget = """          <div className="w-full flex justify-center mt-12 mb-8">
            <Button
              onPress={() => setIsModalOpen(true)}
              variant="secondary" className=" rounded-xl flex items-center gap-2 px-6 py-6 transition-all cursor-pointer"
            >
              <Plus size={20} /> Add New Widget
            </Button>
          </div>"""

new_add_widget = """          <div className="w-full flex justify-center mt-12 mb-8">
            <Button
              onPress={() => setIsModalOpen(true)}
              variant="bordered"
              className="rounded-xl flex items-center gap-2 px-6 py-6 transition-all cursor-pointer bg-white/5 border-white/10 hover:bg-white/10 text-foreground/70 hover:text-foreground shadow-lg backdrop-blur-md"
            >
              <Plus size={20} /> Add New Widget
            </Button>
          </div>"""
content = content.replace(old_add_widget, new_add_widget)

# Fix inside Add Widget modal buttons
old_modal_btns = """                  <Button
                    key={widget.type}
                    variant="primary"
                    onPress={() => handleAddManualWidget(widget.type)}
                    className="w-full h-auto text-left flex justify-start items-center gap-4 p-4 rounded-2xl hover:bg-secondary transition-all border border-transparent hover:border-border group cursor-pointer bg-transparent"
                  >"""

new_modal_btns = """                  <Button
                    key={widget.type}
                    variant="ghost"
                    onPress={() => handleAddManualWidget(widget.type)}
                    className="w-full h-auto text-left flex justify-start items-center gap-4 p-4 rounded-2xl hover:bg-white/5 transition-all border border-transparent hover:border-white/10 group cursor-pointer bg-transparent text-foreground"
                  >"""
content = content.replace(old_modal_btns, new_modal_btns)

with open("app/(protected)/home/DashboardClient.tsx", "w") as f:
    f.write(content)
