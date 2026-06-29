import React from "react";
import {
  Plus,
  History,
  MessageSquare,
  Clock,
  MoreVertical,
  Trash,
  Pencil,
} from "lucide-react";
import {
  Button,
  ScrollShadow,
  Text,
  Skeleton,
  Dropdown,
  Label,
  Input,
  Modal,
} from "@heroui/react";
import { cn } from "@/lib/utils";

export interface BudAIChatSession {
  session_id: string;
  title: string;
  last_updated: string;
}

export const AdvisorSidebar = ({
  sessions,
  sessionsLoading,
  activeSessionId,
  router,
  handleNewChat,
  handleDeleteSession,
  handleRenameSession,
}: {
  sessions: BudAIChatSession[];
  sessionsLoading: boolean;
  activeSessionId: string;
  router: ReturnType<typeof import("next/navigation").useRouter>;
  handleNewChat: () => void;
  handleDeleteSession: (id: string) => void;
  handleRenameSession: (id: string, newTitle: string) => Promise<void>;
}) => {
  const [editingId, setEditingId] = React.useState<string | null>(null);
  const [editValue, setEditValue] = React.useState("");

  const startEditing = (s: BudAIChatSession) => {
    setEditingId(s.session_id);
    setEditValue(s.title);
  };

  const saveEdit = async () => {
    if (editingId && editValue.trim()) {
      await handleRenameSession(editingId, editValue);
      setEditingId(null);
    }
  };

  return (
    <aside className="w-full md:w-64 h-full bg-background border-r border-border flex flex-col shrink-0">
      <div className="p-8">
        <Button
          onPress={handleNewChat}
          variant="primary"
          className="w-full h-12 bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20 hover:border-primary/40 rounded-xl transition-all shadow-[0_0_20px_rgba(0,127,255,0.1)] hover:shadow-[0_0_30px_rgba(0,127,255,0.2)] group"
        >
          <div className="flex items-center justify-center gap-2">
            <Plus
              size={16}
              className="group-hover:rotate-90 transition-transform duration-300"
            />
            <span className="font-bold text-[11px] uppercase tracking-widest">
              New Session
            </span>
          </div>
        </Button>
      </div>

      <div className="px-8 pb-4">
        <div className="flex items-center gap-2 mb-4">
          <History size={12} className="text-foreground/40" />
          <Text className="text-[9px] font-bold tracking-[0.2em] uppercase text-foreground/40">
            Recent Analysis
          </Text>
        </div>
      </div>

      <ScrollShadow className="flex-1 px-4 pb-8 overflow-y-auto h-[calc(100vh-250px)]">
        {sessionsLoading ? (
          <div className="flex flex-col gap-3 px-4">
            {[1, 2, 3].map((i) => (
              <Skeleton
                key={i}
                className="w-full h-16 rounded-xl bg-content1/50"
              />
            ))}
          </div>
        ) : sessions.length === 0 ? (
          <div className="px-4 py-8 text-center border border-dashed border-border rounded-xl bg-content1/20 mx-4">
            <MessageSquare
              size={20}
              className="text-foreground/20 mx-auto mb-3"
            />
            <Text className="text-[10px] text-foreground/40 uppercase tracking-widest font-bold">
              No History
            </Text>
          </div>
        ) : (
          sessions.map((s) => (
            <div
              key={s.session_id}
              className={cn(
                "group relative flex items-center justify-between p-4 mb-2 rounded-xl cursor-pointer transition-all duration-300 border",
                activeSessionId === s.session_id
                  ? "bg-primary/10 border-primary/30 shadow-[0_0_15px_rgba(0,127,255,0.1)]"
                  : "bg-transparent border-transparent hover:bg-content1 hover:border-border",
              )}
            >
              <div
                className="flex flex-col flex-1 min-w-0"
                onClick={() => router.push(`/advisor?session=${s.session_id}`)}
              >
                <Text
                  className={cn(
                    "text-xs font-semibold truncate mb-1.5 transition-colors",
                    activeSessionId === s.session_id
                      ? "text-primary"
                      : "text-foreground/80 group-hover:text-foreground",
                  )}
                >
                  {s.title || "Financial Analysis"}
                </Text>
                <div className="flex items-center gap-2">
                  <Clock
                    size={10}
                    className={
                      activeSessionId === s.session_id
                        ? "text-primary/60"
                        : "text-foreground/30"
                    }
                  />
                  <Text
                    className={cn(
                      "text-[9px] font-bold uppercase tracking-widest truncate",
                      activeSessionId === s.session_id
                        ? "text-primary/60"
                        : "text-foreground/30",
                    )}
                  >
                    {new Date(s.last_updated).toLocaleDateString()}
                  </Text>
                </div>
              </div>

              <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                <Dropdown>
                  <Button
                    isIconOnly
                    size="sm"
                    variant="primary"
                    className="text-foreground/50 hover:text-foreground h-6 w-6 min-w-6 cursor-pointer data-[hover=true]:bg-transparent focus:outline-none data-[focus-visible=true]:outline-none data-[focus-visible=true]:ring-0"
                  >
                    <MoreVertical size={14} />
                  </Button>
                  <Dropdown.Popover
                    placement="bottom end"
                    className="bg-background/90 backdrop-blur-2xl border border-border/50 rounded-xl min-w-40 shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200 p-1"
                  >
                    <Dropdown.Menu
                      aria-label="Session Actions"
                      className="flex flex-col gap-1 w-full"
                      onAction={(key) => {
                        if (key === "rename") startEditing(s);
                        if (key === "delete") handleDeleteSession(s.session_id);
                      }}
                    >
                      <Dropdown.Item
                        id="rename"
                        textValue="Rename Session"
                        className="text-foreground hover:bg-secondary data-[hover=true]:bg-secondary transition-colors w-full flex items-center px-3 py-2 rounded-lg"
                      >
                        <div className="flex items-center gap-3">
                          <Pencil size={14} />
                          <Label className="text-xs font-medium cursor-pointer">
                            Rename Session
                          </Label>
                        </div>
                      </Dropdown.Item>

                      <Dropdown.Item
                        id="delete"
                        textValue="Delete Session"
                        variant="danger"
                        className="text-danger hover:bg-danger/10 data-[hover=true]:bg-danger/10 transition-colors w-full flex items-center px-3 py-2 rounded-lg"
                      >
                        <div className="flex items-center gap-3">
                          <Trash size={14} />
                          <Label className="text-xs font-medium cursor-pointer">
                            Delete Session
                          </Label>
                        </div>
                      </Dropdown.Item>
                    </Dropdown.Menu>
                  </Dropdown.Popover>
                </Dropdown>
              </div>
            </div>
          ))
        )}
      </ScrollShadow>

      <Modal>
        <Modal.Backdrop
          isOpen={editingId !== null}
          onOpenChange={(open) => {
            if (!open) setEditingId(null);
          }}
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm animate-in fade-in duration-200"
        >
          <Modal.Container className="relative w-full max-w-md bg-content1 border border-border rounded-3xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200 mx-4">
            <Modal.Dialog>
              <Modal.CloseTrigger
                slot="close"
                className="absolute right-4 top-4 text-foreground/50 hover:text-foreground transition-colors z-10"
              />
              <Modal.Header className="border-b border-border px-6 py-4 pr-12">
                <Modal.Heading className="text-sm font-bold uppercase tracking-widest text-foreground">
                  Rename Session
                </Modal.Heading>
              </Modal.Header>
              <Modal.Body className="p-6">
                <Input
                  autoFocus
                  placeholder="Enter new title..."
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") saveEdit();
                  }}
                  variant="primary"
                  className="w-full h-10 text-foreground bg-transparent focus-within:bg-content2 transition-colors rounded-md pl-1"
                />
              </Modal.Body>
              <Modal.Footer className="border-t border-border px-6 py-4 flex justify-end gap-3">
                <Button
                  slot="close"
                  variant="primary"
                  className="text-muted-foreground hover:bg-secondary hover:text-red-500 transition-all border-none h-10 rounded-xl px-4 text-xs font-semibold"
                >
                  Cancel
                </Button>
                <Button
                  slot="close"
                  onPress={saveEdit}
                  className="bg-primary text-primary-foreground font-black uppercase tracking-widest text-[11px] h-10 rounded-xl px-6 shadow-[0_0_15px_rgba(0,127,255,0.3)] hover:shadow-[0_0_25px_rgba(0,127,255,0.5)] transition-all border-none cursor-pointer"
                >
                  Save Changes
                </Button>
              </Modal.Footer>
            </Modal.Dialog>
          </Modal.Container>
        </Modal.Backdrop>
      </Modal>
    </aside>
  );
};
