"use client";

import { ToastContainer } from "react-toastify";
import { ChakraProvider } from "@chakra-ui/react";
import { QueryClient, QueryClientProvider } from "react-query";
import { Client } from "@langchain/langgraph-sdk";

import { ChatWindow } from "../components/ChatWindow";
import { LangGraphClientContext } from "../hooks/useLangGraphClient";
import {useTranslations} from 'next-intl';

const apiUrl = process.env.NEXT_PUBLIC_API_URL
  ? process.env.NEXT_PUBLIC_API_URL
  : "http://localhost:3000/api";

const apiKey = process.env.NEXT_PUBLIC_LANGCHAIN_API_KEY;
const apiKeywithnext = process.env.LANGCHAIN_API_KEY;

export default function Home() {
  const queryClient = new QueryClient();
  const t = useTranslations('HomePage'); 
  
  const langGraphClient = new Client({
  apiUrl,
  apiKey: process.env.NEXT_PUBLIC_LANGCHAIN_API_KEY,
});



  
  return (
    <LangGraphClientContext.Provider value={langGraphClient}>
      <QueryClientProvider client={queryClient}>
        <ChakraProvider>
          <ToastContainer />
          <ChatWindow />
        </ChakraProvider>
      </QueryClientProvider>
    </LangGraphClientContext.Provider>
  );
}
