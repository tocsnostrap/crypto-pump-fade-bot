import { useState, useRef, useEffect, useCallback } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  MessageSquare,
  Send,
  Plus,
  Trash2,
  Settings,
  Loader2,
  ImagePlus,
  X,
} from "lucide-react";
import type { Conversation, Message } from "@shared/schema";

interface ConversationWithMessages extends Conversation {
  messages: Message[];
}

interface StreamEvent {
  content?: string;
  done?: boolean;
  error?: string;
  config_changed?: boolean;
  changes?: Record<string, { old: unknown; new: unknown }>;
}

function formatMessageContent(content: string): string {
  return content.replace(/```CONFIG_CHANGE\s*\n[\s\S]*?\n```/g, "").trim();
}

function extractImageFromContent(content: string): { text: string; imageUrl: string | null } {
  const match = content.match(/\[Image attached: (data:image\/[^\]]+)\]/);
  if (match) {
    return {
      text: content.replace(match[0], "").trim(),
      imageUrl: match[1],
    };
  }
  return { text: content, imageUrl: null };
}

function ConfigChangeBadge({ changes }: { changes: Record<string, { old: unknown; new: unknown }> }) {
  return (
    <div className="mt-2 p-3 rounded-md bg-primary/10 border border-primary/20">
      <div className="flex items-center gap-2 mb-2">
        <Settings className="h-4 w-4 text-primary" />
        <span className="text-sm font-medium">Config Updated</span>
      </div>
      <div className="space-y-1">
        {Object.entries(changes).map(([key, val]) => (
          <div key={key} className="text-xs font-mono text-muted-foreground">
            <span className="text-foreground">{key}</span>:{" "}
            <span className="text-destructive line-through">{JSON.stringify(val.old)}</span>{" "}
            <span className="text-profit">{JSON.stringify(val.new)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function MessageBubble({ message, configChanges }: { message: Message; configChanges?: Record<string, { old: unknown; new: unknown }> | null }) {
  const isUser = message.role === "user";
  const { text, imageUrl } = extractImageFromContent(message.content);
  const displayContent = isUser ? text : formatMessageContent(text);

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`} data-testid={`message-${message.id}`}>
      <div className={`max-w-[85%] rounded-lg px-4 py-2.5 text-sm ${isUser ? "bg-primary text-primary-foreground" : "bg-muted"}`}>
        {imageUrl && (
          <img
            src={imageUrl}
            alt="Attached"
            className="max-w-full max-h-48 rounded-md mb-2 object-contain"
            data-testid={`img-attachment-${message.id}`}
          />
        )}
        {displayContent && <div className="whitespace-pre-wrap break-words">{displayContent}</div>}
        {configChanges && <ConfigChangeBadge changes={configChanges} />}
      </div>
    </div>
  );
}

function StreamingBubble({ text }: { text: string }) {
  const displayContent = formatMessageContent(text);
  return (
    <div className="flex justify-start mb-3">
      <div className="max-w-[85%] rounded-lg px-4 py-2.5 text-sm bg-muted">
        <div className="whitespace-pre-wrap break-words">{displayContent || <Loader2 className="h-4 w-4 animate-spin" />}</div>
      </div>
    </div>
  );
}

const MAX_IMAGE_SIZE = 4 * 1024 * 1024;

export default function BotChat() {
  const [activeConversationId, setActiveConversationId] = useState<number | null>(null);
  const [input, setInput] = useState("");
  const [streamingText, setStreamingText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [configChanges, setConfigChanges] = useState<Record<string, { old: unknown; new: unknown }> | null>(null);
  const [pendingImage, setPendingImage] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { data: conversations = [] } = useQuery<Conversation[]>({
    queryKey: ["/api/conversations"],
  });

  useEffect(() => {
    if (!activeConversationId && conversations.length > 0) {
      setActiveConversationId(conversations[0].id);
    }
  }, [conversations, activeConversationId]);

  const { data: activeConversation, refetch: refetchConversation } = useQuery<ConversationWithMessages>({
    queryKey: ["/api/conversations", activeConversationId],
    enabled: !!activeConversationId,
  });

  const createConversationMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("POST", "/api/conversations", { title: "New Chat" });
      return res.json() as Promise<Conversation>;
    },
    onSuccess: (conv) => {
      queryClient.invalidateQueries({ queryKey: ["/api/conversations"] });
      setActiveConversationId(conv.id);
    },
  });

  const deleteConversationMutation = useMutation({
    mutationFn: async (id: number) => {
      await apiRequest("DELETE", `/api/conversations/${id}`);
      return id;
    },
    onSuccess: (deletedId) => {
      queryClient.invalidateQueries({ queryKey: ["/api/conversations"] });
      if (activeConversationId === deletedId) {
        const remaining = conversations.filter((c) => c.id !== deletedId);
        setActiveConversationId(remaining.length > 0 ? remaining[0].id : null);
      }
    },
  });

  const scrollToBottom = useCallback(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [activeConversation?.messages, streamingText, scrollToBottom]);

  const handleImageFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    if (file.size > MAX_IMAGE_SIZE) {
      alert("Image must be under 4MB");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      setPendingImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  }, []);

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of Array.from(items)) {
      if (item.type.startsWith("image/")) {
        e.preventDefault();
        const file = item.getAsFile();
        if (file) handleImageFile(file);
        return;
      }
    }
  }, [handleImageFile]);

  const sendMessage = useCallback(async () => {
    if ((!input.trim() && !pendingImage) || !activeConversationId || isStreaming) return;

    const userMessage = input.trim();
    const imageData = pendingImage;
    setInput("");
    setPendingImage(null);
    setIsStreaming(true);
    setStreamingText("");
    setConfigChanges(null);

    try {
      const response = await fetch(`/api/conversations/${activeConversationId}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: userMessage,
          image: imageData || undefined,
        }),
      });

      if (!response.ok) throw new Error("Failed to send message");

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response stream");

      const decoder = new TextDecoder();
      let buffer = "";
      let accumulated = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event: StreamEvent = JSON.parse(line.slice(6));
            if (event.content) {
              accumulated += event.content;
              setStreamingText(accumulated);
            }
            if (event.config_changed && event.changes) {
              setConfigChanges(event.changes);
              queryClient.invalidateQueries({ queryKey: ["/api/config"] });
            }
            if (event.done) {
              setIsStreaming(false);
              setStreamingText("");
              refetchConversation();
              queryClient.invalidateQueries({ queryKey: ["/api/conversations"] });
            }
            if (event.error) {
              setIsStreaming(false);
              setStreamingText("");
            }
          } catch {}
        }
      }
    } catch (err) {
      console.error("Chat error:", err);
      setIsStreaming(false);
      setStreamingText("");
    }
  }, [input, pendingImage, activeConversationId, isStreaming, refetchConversation]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const messages = activeConversation?.messages || [];

  return (
    <Card className="flex flex-col h-[400px] md:h-[600px]">
      <CardHeader className="flex flex-row items-center justify-between gap-2 pb-3 shrink-0">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5 text-primary" />
          <CardTitle className="text-lg">Strategy Chat</CardTitle>
        </div>
        <div className="flex items-center gap-1">
          {activeConversationId && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => deleteConversationMutation.mutate(activeConversationId)}
              data-testid="button-delete-chat"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => createConversationMutation.mutate()}
            disabled={createConversationMutation.isPending}
            data-testid="button-new-chat"
          >
            <Plus className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>

      {conversations.length > 1 && (
        <div className="px-4 pb-2 shrink-0">
          <div className="flex flex-wrap gap-1">
            {conversations.slice(0, 5).map((conv) => (
              <Badge
                key={conv.id}
                variant={conv.id === activeConversationId ? "default" : "outline"}
                className="cursor-pointer"
                onClick={() => setActiveConversationId(conv.id)}
                data-testid={`badge-conversation-${conv.id}`}
              >
                {conv.title}
              </Badge>
            ))}
          </div>
        </div>
      )}

      <CardContent className="flex-1 flex flex-col min-h-0 p-4 pt-0 gap-3">
        <div ref={scrollRef} className="flex-1 overflow-y-auto pr-1">
          {!activeConversationId ? (
            <div className="h-full flex flex-col items-center justify-center text-center gap-3 p-4">
              <MessageSquare className="h-10 w-10 text-muted-foreground/40" />
              <div>
                <p className="text-sm font-medium">Ask me about your trades</p>
                <p className="text-xs text-muted-foreground mt-1">
                  I can analyze performance, suggest strategy changes, and modify bot parameters.
                </p>
              </div>
              <Button
                variant="outline"
                onClick={() => createConversationMutation.mutate()}
                disabled={createConversationMutation.isPending}
                data-testid="button-start-chat"
              >
                Start a conversation
              </Button>
            </div>
          ) : messages.length === 0 && !isStreaming ? (
            <div className="h-full flex flex-col items-center justify-center text-center gap-2 p-4">
              <p className="text-sm text-muted-foreground">Try asking:</p>
              <div className="flex flex-col gap-1.5">
                {[
                  "How are my trades performing?",
                  "Should I tighten my stop loss?",
                  "Lower the RSI threshold to 65",
                ].map((suggestion) => (
                  <Button
                    key={suggestion}
                    variant="outline"
                    className="text-xs h-auto py-1.5 px-3"
                    onClick={() => {
                      setInput(suggestion);
                      textareaRef.current?.focus();
                    }}
                    data-testid={`button-suggestion-${suggestion.slice(0, 10)}`}
                  >
                    {suggestion}
                  </Button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-0">
              {messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}
              {isStreaming && <StreamingBubble text={streamingText} />}
              {configChanges && !isStreaming && (
                <div className="flex justify-start mb-3">
                  <div className="max-w-[85%]">
                    <ConfigChangeBadge changes={configChanges} />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {activeConversationId && (
          <div className="shrink-0 space-y-2">
            {pendingImage && (
              <div className="relative inline-block">
                <img
                  src={pendingImage}
                  alt="Preview"
                  className="max-h-24 rounded-md border"
                  data-testid="img-preview"
                />
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-destructive text-destructive-foreground"
                  onClick={() => setPendingImage(null)}
                  data-testid="button-remove-image"
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
            )}
            <div className="flex gap-2">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleImageFile(file);
                  e.target.value = "";
                }}
                data-testid="input-file-upload"
              />
              <Button
                variant="ghost"
                size="icon"
                onClick={() => fileInputRef.current?.click()}
                disabled={isStreaming}
                data-testid="button-attach-image"
              >
                <ImagePlus className="h-4 w-4" />
              </Button>
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                onPaste={handlePaste}
                placeholder="Ask about trades, strategy, or paste an image..."
                className="resize-none min-h-[40px] max-h-[100px] flex-1"
                rows={1}
                disabled={isStreaming}
                data-testid="input-chat-message"
              />
              <Button
                size="icon"
                onClick={sendMessage}
                disabled={(!input.trim() && !pendingImage) || isStreaming}
                data-testid="button-send-message"
              >
                {isStreaming ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
