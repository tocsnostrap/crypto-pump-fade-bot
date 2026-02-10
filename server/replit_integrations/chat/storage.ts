import { queryDB } from "../../db-helper";

export interface ChatConversation {
  id: number;
  title: string;
  created_at: string;
}

export interface ChatMessage {
  id: number;
  conversation_id: number;
  role: string;
  content: string;
  created_at: string;
}

export interface IChatStorage {
  getConversation(id: number): Promise<ChatConversation | undefined>;
  getAllConversations(): Promise<ChatConversation[]>;
  createConversation(title: string): Promise<ChatConversation>;
  deleteConversation(id: number): Promise<void>;
  getMessagesByConversation(conversationId: number): Promise<ChatMessage[]>;
  createMessage(conversationId: number, role: string, content: string): Promise<ChatMessage>;
}

export const chatStorage: IChatStorage = {
  async getConversation(id: number) {
    const rows = await queryDB("SELECT * FROM conversations WHERE id = $1", [id]);
    return rows[0] as ChatConversation | undefined;
  },

  async getAllConversations() {
    return await queryDB("SELECT * FROM conversations ORDER BY created_at DESC") as ChatConversation[];
  },

  async createConversation(title: string) {
    const rows = await queryDB(
      "INSERT INTO conversations (title) VALUES ($1) RETURNING *",
      [title]
    );
    return rows[0] as ChatConversation;
  },

  async deleteConversation(id: number) {
    await queryDB("DELETE FROM messages WHERE conversation_id = $1", [id]);
    await queryDB("DELETE FROM conversations WHERE id = $1", [id]);
  },

  async getMessagesByConversation(conversationId: number) {
    return await queryDB(
      "SELECT * FROM messages WHERE conversation_id = $1 ORDER BY created_at ASC",
      [conversationId]
    ) as ChatMessage[];
  },

  async createMessage(conversationId: number, role: string, content: string) {
    const rows = await queryDB(
      "INSERT INTO messages (conversation_id, role, content) VALUES ($1, $2, $3) RETURNING *",
      [conversationId, role, content]
    );
    return rows[0] as ChatMessage;
  },
};
