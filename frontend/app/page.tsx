"use client";

import React, { useState } from "react";
import {
  AppendMessage,
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
} from "@assistant-ui/react";
import { v4 as uuidv4 } from "uuid";
import { MyThread } from "./components/Primitives";
import { useExternalMessageConverter } from "@assistant-ui/react";
import { BaseMessage, HumanMessage } from "@langchain/core/messages";
import { Toaster } from "./components/ui/toaster";
import {
  convertLangchainMessages,
  convertToOpenAIFormat,
} from "./utils/convert_messages";
import { useGraph } from "./hooks/useGraph";
import { ThreadHistory } from "./components/ThreadHistory";
import { useUser } from "./hooks/useUser";
import { useThreads } from "./hooks/useThreads";
import { useToast } from "./hooks/use-toast";
import { SelectModel } from "./components/SelectModel";

export default function ContentComposerChatInterface(): React.ReactElement {
  const { toast } = useToast();
  const { userId } = useUser();
  const {
    userThreads,
    getUserThreads,
    isUserThreadsLoading,
    createThread,
    threadId: currentThread,
    deleteThread,
  } = useThreads(userId);
  const {
    messages,
    setMessages,
    streamMessage,
    assistantId,
    switchSelectedThread,
    selectedModel,
    setSelectedModel,
  } = useGraph({
    userId,
    threadId: currentThread,
  });
  const [isRunning, setIsRunning] = useState(false);
  const [selectedPage, setSelectedPage] = useState<"chat" | "about" | "login">("chat");

  const isSubmitDisabled = !userId || !assistantId || !currentThread;

  async function onNew(message: AppendMessage): Promise<void> {
    if (isSubmitDisabled) {
      let description = "";
      if (!userId) {
        description = "Unable to find user ID. Please try again later.";
      } else if (!assistantId) {
        description = "Unable to find assistant ID. Please try again later.";
      } else if (!currentThread) {
        description =
          "Unable to find current thread ID. Please try again later.";
      }
      toast({
        title: "Failed to send message",
        description,
      });
      return;
    }
    if (message.content[0]?.type !== "text") {
      throw new Error("Only text messages are supported");
    }
    setIsRunning(true);

    try {
      const humanMessage = new HumanMessage({
        content: message.content[0].text,
        id: uuidv4(),
      });

      setMessages((prevMessages) => [...prevMessages, humanMessage]);

      await streamMessage({
        messages: [convertToOpenAIFormat(humanMessage)],
      });
    } finally {
      setIsRunning(false);
      // Re-fetch threads so that the current thread's title is updated.
      await getUserThreads(userId);
    }
  }

  const threadMessages = useExternalMessageConverter<BaseMessage>({
    callback: convertLangchainMessages,
    messages: messages,
    isRunning,
  });

  const runtime = useExternalStoreRuntime({
    messages: threadMessages,
    isRunning,
    onNew,
  });

  return (
    <div className="overflow-hidden w-full flex md:flex-row flex-col relative">
      <div>
        <ThreadHistory
          isUserThreadsLoading={isUserThreadsLoading}
          getUserThreads={getUserThreads}
          isEmpty={messages.length === 0}
          switchSelectedThread={switchSelectedThread}
          currentThread={currentThread}
          userThreads={userThreads}
          userId={userId}
          createThread={createThread}
          assistantId={assistantId}
          deleteThread={(id) => deleteThread(id, () => setMessages([]))}
          clearMessages={() => setMessages([])}
          setSelectedPage={setSelectedPage}
        />
      </div>
      <div className="w-full h-full overflow-hidden">
        {selectedPage === "chat" ? (
          <>
            {messages.length > 0 ? (
              <div className="absolute top-4 right-4 z-10">
                <SelectModel
                  selectedModel={selectedModel}
                  setSelectedModel={setSelectedModel}
                />
              </div>
            ) : null}
            <AssistantRuntimeProvider runtime={runtime}>
              <MyThread
                selectedModel={selectedModel}
                setSelectedModel={setSelectedModel}
                submitDisabled={isSubmitDisabled}
                messages={messages}
              />
            </AssistantRuntimeProvider>
          </>
        ) : selectedPage === "about" ? (
          <div className="p-4">
            <h1 className="text-2xl font-bold mb-4">About StockGPT</h1>
            <p>StockGPT is a chatbot for stock trading.</p>
            {/* Add more content for the About page */}
          </div>
        ) : selectedPage === "login" ? (
          <div className="p-4">
            <h1 className="text-2xl font-bold mb-4">Login</h1>
            {/* Add login form or content */}
          </div>
        ) : null}
      </div>
      <Toaster />
    </div>
  );
}