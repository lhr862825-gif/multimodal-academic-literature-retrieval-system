"use client"

import type React from "react"

import { useState, useRef, useEffect, useMemo } from "react"
import axios from "axios"
import ReactMarkdown from "react-markdown"
import { encode as encodeCl100k } from "gpt-tokenizer/encoding/cl100k_base"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible"
import { MessageSquare, Settings, Paperclip, ImageIcon, Send, Plus, Sparkles, X, RotateCcw, ThumbsUp, ThumbsDown, BarChart2, Activity, User, Home, Hash, Globe, ChevronRight, Loader2 } from "lucide-react"
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid } from "recharts"
import { HistoryItem } from "@/components/ui/history-item"
import { Switch } from "@/components/ui/switch"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { cn } from "@/lib/utils"
import { HaloButton } from "@/components/ui/halo-button"

type Message = {
  id: string
  role: "user" | "assistant"
  content: string
  attachments?: Array<{
    type: "image" | "file"
    name: string
    url: string
  }>
  metadata?: {
    questionId?: string
    retrievalTime?: number
    generationTime?: number
    confidence?: number
    sources?: string[]
    thinking?: string[]
    timeline?: Array<{
      timestamp?: number
      event?: string
      message: string
      detail?: string
    }>
  }
}

type Conversation = {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
  updatedAt: Date
}

export function MultimodalChat() {
  const [activeView, setActiveView] = useState<"chat" | "settings" | "dashboard">("chat")
  const [sidebarTab, setSidebarTab] = useState<"topics" | "settings" | "dashboard">("topics")

  const [conversations, setConversations] = useState<Conversation[]>([
    {
      id: "1",
      title: "Current conversation",
      messages: [
        {
          id: "1",
          role: "assistant",
          content: "Hi! How can I help you today?",
        },
      ],
      createdAt: new Date(),
      updatedAt: new Date(),
    },
  ])
  const [currentConversationId, setCurrentConversationId] = useState("1")

  const currentConversation = conversations.find((c) => c.id === currentConversationId)
  const messages = currentConversation?.messages || []

  const [inputValue, setInputValue] = useState("")
  const [attachments, setAttachments] = useState<
    Array<{
      type: "image" | "file"
      name: string
      url: string
    }>
  >([])
  const [isLoading, setIsLoading] = useState(false)
  const [feedbacks, setFeedbacks] = useState<Record<string, "up" | "down" | undefined>>({})
  const [growthSources, setGrowthSources] = useState({ auto: 0, user: 0, manual: 0 })
  const [effBaseline, setEffBaseline] = useState<null | { avgRetrieval: number; avgGeneration: number; timestamp: string }>(null)
  const [activeSlice, setActiveSlice] = useState<number | null>(null)

  const [selectedModel, setSelectedModel] = useState("Qwen3-8B")
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(2048)
  const [streamEnabled, setStreamEnabled] = useState(true)
  const [maxTokensEnabled, setMaxTokensEnabled] = useState(false)
  const [chainLevel, setChainLevel] = useState("medium") // basic, medium, advanced
  const [chainLength, setChainLength] = useState(6)

  const [growthEvents, setGrowthEvents] = useState<Array<{ type: "auto" | "user" | "manual"; ts: number }>>([])

  const scrollRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const streamIntervalRef = useRef<NodeJS.Timeout | null>(null)

  const prevMessagesLengthRef = useRef(messages.length)

  useEffect(() => {
    const scrollEl = scrollRef.current
    if (!scrollEl) return

    const isNewMessage = messages.length > prevMessagesLengthRef.current
    prevMessagesLengthRef.current = messages.length

    // Define a threshold for "being at the bottom"
    const isAtBottom = scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight < 150

    if (isNewMessage || isAtBottom) {
      // Use setTimeout to ensure DOM is updated
      setTimeout(() => {
        scrollEl.scrollTo({ top: scrollEl.scrollHeight, behavior: "smooth" })
      }, 100)
    }
  }, [messages])

  useEffect(() => {
    try {
      const saved = localStorage.getItem("effBaseline")
      if (saved) setEffBaseline(JSON.parse(saved))
      const savedEvents = localStorage.getItem("growthEvents")
      if (savedEvents) {
        const parsed = JSON.parse(savedEvents)
        if (Array.isArray(parsed)) setGrowthEvents(parsed)
      }
    } catch {}
  }, [])

  const isDefaultGreeting = (text: string) => text.trim() === "Hi! How can I help you today?"

  const handleFeedback = (messageId: string, rating: "up" | "down") => {
    setFeedbacks((prev) => ({ ...prev, [messageId]: rating }))
  }

  const analytics = useMemo(() => {
    const all = conversations.flatMap((c) => c.messages)
    const assistants = all.filter((m) => m.role === "assistant" && !isDefaultGreeting(m.content))
    const avg = (arr: number[]) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0)
    const retrievals = assistants.map((m) => m.metadata?.retrievalTime || 0).filter((v) => v > 0)
    const generations = assistants.map((m) => m.metadata?.generationTime || 0).filter((v) => v > 0)
    const completionTokens = assistants.map((m) => encodeCl100k(m.content).length).filter((v) => v > 0)
    const up = Object.entries(feedbacks).filter(([, v]) => v === "up").length
    const down = Object.entries(feedbacks).filter(([, v]) => v === "down").length
    const sourceMap: Record<string, number> = {}
    assistants.forEach((m) => {
      const src = m.metadata?.sources || []
      src.forEach((s) => {
        sourceMap[s] = (sourceMap[s] || 0) + 1
      })
    })
    const sources = Object.entries(sourceMap)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
    return {
      totalMessages: all.length,
      assistantMessages: assistants.length,
      avgRetrieval: avg(retrievals),
      avgGeneration: avg(generations),
      avgCompletionTokens: avg(completionTokens),
      up,
      down,
      sources,
    }
  }, [conversations, feedbacks])

  const efficiencySeries = useMemo(() => {
    const all = conversations.flatMap((c) => c.messages)
    const assistants = all.filter((m) => m.role === "assistant" && !isDefaultGreeting(m.content))
    const baseR = effBaseline?.avgRetrieval || 0
    const baseG = effBaseline?.avgGeneration || 0
    return assistants
      .filter((m) => typeof m.metadata?.retrievalTime === "number" || typeof m.metadata?.generationTime === "number")
      .map((m, i) => {
        const rt = m.metadata?.retrievalTime || 0
        const gt = m.metadata?.generationTime || 0
        const rGain = baseR > 0 ? ((baseR - rt) / baseR) * 100 : null
        const gGain = baseG > 0 ? ((baseG - gt) / baseG) * 100 : null
        return {
          ts: Number(m.id) || Date.now(),
          index: i + 1,
          retrievalGain: rGain,
          generationGain: gGain,
          retrievalTime: rt,
          generationTime: gt,
        }
      })
  }, [conversations, effBaseline])

  const copyAssistantToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text)
    } catch {}
  }

  const saveEfficiencyBaseline = () => {
    const b = { avgRetrieval: analytics.avgRetrieval, avgGeneration: analytics.avgGeneration, timestamp: new Date().toISOString() }
    setEffBaseline(b)
    try {
      localStorage.setItem("effBaseline", JSON.stringify(b))
    } catch {}
  }

  const pushGrowthEvent = (type: "auto" | "user" | "manual", count = 1) => {
    const now = Date.now()
    const events = Array.from({ length: count }, () => ({ type, ts: now }))
    setGrowthEvents((prev) => {
      const next = [...prev, ...events]
      try {
        localStorage.setItem("growthEvents", JSON.stringify(next))
      } catch {}
      return next
    })
  }

  const weeklyCounts = useMemo(() => {
    const weekAgo = Date.now() - 7 * 24 * 3600 * 1000
    const counts = { auto: 0, user: 0, manual: 0 }
    growthEvents.forEach((e) => {
      if (e.ts >= weekAgo) counts[e.type] += 1
    })
    return counts
  }, [growthEvents])

  const lighten = (hex: string, ratio: number) => {
    const h = hex.replace("#", "")
    const bigint = parseInt(h, 16)
    const r = (bigint >> 16) & 255
    const g = (bigint >> 8) & 255
    const b = bigint & 255
    const nr = Math.round(r * (1 - ratio) + 255 * ratio)
    const ng = Math.round(g * (1 - ratio) + 255 * ratio)
    const nb = Math.round(b * (1 - ratio) + 255 * ratio)
    const toHex = (v: number) => v.toString(16).padStart(2, "0")
    return `#${toHex(nr)}${toHex(ng)}${toHex(nb)}`
  }

  const growthData = useMemo(
    () => [
      { key: "auto" as const, name: "Auto learning", value: growthSources.auto, color: "#547CAE" },
      { key: "user" as const, name: "User interaction", value: growthSources.user, color: "#82AADB" },
      { key: "manual" as const, name: "Manual curation", value: growthSources.manual, color: "#CAD8D9" },
    ],
    [growthSources],
  )
  const totalGrowth = useMemo(() => growthData.reduce((s, d) => s + d.value, 0), [growthData])

  const regenerateAnswer = async (assistantMessageId: string) => {
    const conv = conversations.find((c) => c.id === currentConversationId)
    if (!conv) return
    const idx = conv.messages.findIndex((m) => m.id === assistantMessageId)
    let userMsg: Message | undefined
    for (let i = idx - 1; i >= 0; i--) {
      if (conv.messages[i].role === "user") {
        userMsg = conv.messages[i]
        break
      }
    }
    if (!userMsg) {
      const lastUser = [...conv.messages].reverse().find((m) => m.role === "user")
      if (!lastUser) return
      userMsg = lastUser
    }

    setIsLoading(true)
    try {
      const questionResponse = await fetch("/api/chat/question", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMsg.content,
          conversationId: currentConversationId,
          attachments: (userMsg.attachments || []).map((a) => ({ type: a.type, name: a.name })),
          stream: streamEnabled,
          maxTokens: maxTokensEnabled ? maxTokens : undefined,
          chainLength,
        }),
      })
      const questionData = await questionResponse.json()

      const retrieveResponse = await fetch("/api/chat/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          questionId: questionData.data.questionId,
          query: userMsg.content,
        }),
      })
      const retrieveData = await retrieveResponse.json()

      const generateResponse = await fetch("/api/chat/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          questionId: questionData.data.questionId,
          query: userMsg.content,
          retrievedDocs: retrieveData.data.results,
          conversationHistory: conv.messages.slice(-5),
          chainLength,
        }),
      })
      const generateData = await generateResponse.json()

      const meta = {
        questionId: questionData.data.questionId,
        retrievalTime: retrieveData.data.retrievalTime,
        generationTime: generateData.data.generationTime,
        confidence: generateData.data.confidence,
        sources: generateData.data.sources,
        thinking: generateData.data.thinking,
      }

      setConversations((prev) =>
        prev.map((c) =>
          c.id === currentConversationId
            ? {
                ...c,
                messages: c.messages.map((m) =>
                  m.id === assistantMessageId
                    ? { ...m, content: streamEnabled ? "" : generateData.data.answer, metadata: meta }
                    : m,
                ),
                updatedAt: new Date(),
              }
            : c,
        ),
      )

      setGrowthSources((s) => ({ ...s, auto: s.auto + 1 }))
      pushGrowthEvent("auto", 1)

      if (streamEnabled) {
        const full = generateData.data.answer as string
        let i = 0
        const step = 8
        if (streamIntervalRef.current) clearInterval(streamIntervalRef.current)
        streamIntervalRef.current = setInterval(() => {
          i += step
          const slice = full.slice(0, i)
          setConversations((prev) =>
            prev.map((c) =>
              c.id === currentConversationId
                ? {
                    ...c,
                    messages: c.messages.map((m) => (m.id === assistantMessageId ? { ...m, content: slice } : m)),
                    updatedAt: new Date(),
                  }
                : c,
            ),
          )
          if (i >= full.length) {
            if (streamIntervalRef.current) {
              clearInterval(streamIntervalRef.current)
              streamIntervalRef.current = null
            }
          }
        }, 30)
      }
    } catch (e) {
      setConversations((prev) =>
        prev.map((c) =>
          c.id === currentConversationId
            ? {
                ...c,
                messages: c.messages.map((m) =>
                  m.id === assistantMessageId ? { ...m, content: "Sorry, regeneration failed. Please try again later." } : m,
                ),
                updatedAt: new Date(),
              }
            : c,
        ),
      )
    } finally {
      setIsLoading(false)
    }
  }

  const defaultMaxTokens = useMemo(() => {
    if (selectedModel === "GPT-4 Turbo") return 4096
    return 4096
  }, [selectedModel])

  const inputTokens = useMemo(() => {
    if (!inputValue) return 0
    if (selectedModel === "GPT-4 Turbo") return encodeCl100k(inputValue).length
    return encodeCl100k(inputValue).length
  }, [inputValue, selectedModel])

  const maxTokensDisplay = maxTokensEnabled ? maxTokens : defaultMaxTokens

  const handleSend = async () => {
    if (!inputValue.trim() && attachments.length === 0) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: inputValue,
      attachments: attachments.length > 0 ? [...attachments] : undefined,
    }

    // Update the current conversation's messages
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === currentConversationId
          ? {
              ...conv,
              messages: [...conv.messages, userMessage],
              title: conv.messages.length === 1 ? inputValue.slice(0, 30) : conv.title,
              updatedAt: new Date(),
            }
          : conv,
      ),
    )

    const attCountForGrowth = attachments.length
    setGrowthSources((s) => ({ ...s, user: s.user + 1, manual: s.manual + attCountForGrowth }))
    pushGrowthEvent("user", 1)
    if (attCountForGrowth > 0) pushGrowthEvent("manual", attCountForGrowth)

    const currentInput = inputValue
    setInputValue("")
    setAttachments([])
    setIsLoading(true)

    try {
      // 1. Submit the request
      const { data: submitRes } = await axios.post("/api/v1/query", {
        query: currentInput,
        // if there are files, this is where upload logic could be extended
      })

      if (!submitRes.success) throw new Error(submitRes.message || "Submission failed")
      const taskId = submitRes.data.taskId

      // Create a temporary message for live status updates
      const tempMsgId = Date.now().toString()
      
      // Immediately show a "thinking" message
      setConversations((prev) =>
        prev.map((conv) =>
          conv.id === currentConversationId
            ? {
                ...conv,
                messages: [
                  ...conv.messages,
                  {
                    id: tempMsgId,
                    role: "assistant",
                    content: "🤔 Agent is thinking...",
                    metadata: { timeline: [] },
                  },
                ],
                updatedAt: new Date(),
              }
            : conv,
        ),
      )

      // 2. Poll the status
      let isCompleted = false
      let retryCount = 0
      
      while (!isCompleted && retryCount < 120) { // poll at most 120 times
        await new Promise((resolve) => setTimeout(resolve, 1000))
        
        const { data: taskRes } = await axios.get(`/api/v1/task/${taskId}`)
        
        if (taskRes.success && taskRes.data) {
          const { status, result, timeline } = taskRes.data
          
          // Update the timeline display in real time
          if (timeline && timeline.length > 0) {
             setConversations((prev) =>
               prev.map((conv) =>
                 conv.id === currentConversationId
                   ? {
                       ...conv,
                       messages: conv.messages.map((msg) =>
                         msg.id === tempMsgId
                           ? {
                               ...msg,
                               metadata: { ...msg.metadata, timeline: timeline },
                             }
                           : msg
                       ),
                     }
                   : conv,
               ),
             )
          }

          if (status === "completed") {
            isCompleted = true
            
            // Update the final result
            setConversations((prev) =>
              prev.map((conv) =>
                conv.id === currentConversationId
                  ? {
                      ...conv,
                      messages: conv.messages.map((msg) =>
                         msg.id === tempMsgId
                           ? {
                               ...msg,
                               content: result.answer || "No answer received",
                               metadata: {
                                 ...msg.metadata,
                                 timeline: timeline, // make sure the final timeline is preserved
                                 sources: result.sources,
                                 confidence: result.confidence,
                                 generationTime: result.generationTime ?? 0,
                                 retrievalTime: result.retrievalTime ?? 0,
                               },
                             }
                           : msg
                      ),
                      updatedAt: new Date(),
                    }
                  : conv,
              ),
            )
            
            setGrowthSources((s) => ({ ...s, auto: s.auto + 1 }))
            pushGrowthEvent("auto", 1)
          } else if (status === "failed") {
            throw new Error(taskRes.data.error || "Task processing failed")
          }
        }
        retryCount++
      }
      
      if (!isCompleted) throw new Error("Request timed out")

    } catch (error) {
      console.error("Request Error:", error)
      setConversations((prev) =>
        prev.map((conv) =>
          conv.id === currentConversationId
            ? {
                ...conv,
                messages: [
                  ...conv.messages,
                  {
                    id: Date.now().toString(),
                    role: "assistant",
                    content: "Sorry, an error occurred. Please check your connection and try again.",
                  },
                ],
                updatedAt: new Date(),
              }
            : conv,
        ),
      )
    } finally {
      setIsLoading(false)
    }
  }

  const createNewConversation = () => {
    const currentConv = conversations.find((c) => c.id === currentConversationId)
    if (currentConv) {
      const hasUserMessages = currentConv.messages.some((msg) => msg.role === "user")
      if (!hasUserMessages) {
        // user has not sent any messages yet, don't create a new conversation
        return
      }
    }

    const newConversation: Conversation = {
      id: Date.now().toString(),
      title: "New chat",
      messages: [
        {
          id: Date.now().toString(),
          role: "assistant",
          content: "Hi! How can I help you today?",
        },
      ],
      createdAt: new Date(),
      updatedAt: new Date(),
    }
    setConversations((prev) => [newConversation, ...prev])
    setCurrentConversationId(newConversation.id)
  }

  const switchConversation = (conversationId: string) => {
    setCurrentConversationId(conversationId)
  }

  const deleteConversation = (conversationId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (conversations.length === 1) {
      alert("You must keep at least one conversation")
      return
    }
    setConversations((prev) => prev.filter((c) => c.id !== conversationId))
    if (currentConversationId === conversationId) {
      setCurrentConversationId(conversations.find((c) => c.id !== conversationId)?.id || conversations[0].id)
    }
  }

  const handleFileSelect = (type: "file" | "image") => {
    if (type === "file") {
      fileInputRef.current?.click()
    } else {
      imageInputRef.current?.click()
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>, type: "file" | "image") => {
    const files = e.target.files
    if (files && files[0]) {
      const file = files[0]
      setAttachments([
        ...attachments,
        {
          type: type === "image" ? "image" : "file",
          name: file.name,
          url: URL.createObjectURL(file),
        },
      ])
    }
  }

  const removeAttachment = (index: number) => {
    setAttachments(attachments.filter((_, i) => i !== index))
  }

  return (
    <div className="flex h-screen w-full bg-[#FDFDFD] text-zinc-800 font-sans selection:bg-primary/20">
      {/* Sidebar */}
      <aside className="w-[260px] flex-shrink-0 flex flex-col bg-[#FAFAFA] border-r border-zinc-100">
        {/* Home / Search Bar */}
        <div className="h-14 flex items-center px-4 border-b border-zinc-50/50">
          <div className="flex-1 flex items-center gap-2 bg-white border border-zinc-200/60 rounded-xl px-3 py-1.5 shadow-sm hover:shadow transition-all cursor-text group">
            <Home className="w-4 h-4 text-zinc-400 group-hover:text-zinc-600 transition-colors" />
            <span className="text-sm text-zinc-500 font-medium">Home</span>
          </div>
          <Button variant="ghost" size="icon" onClick={createNewConversation} className="ml-2 h-8 w-8 rounded-full hover:bg-zinc-200/50 text-zinc-500">
            <Plus className="w-5 h-5" />
          </Button>
        </div>

        {/* Tabs */}
        <div className="flex items-center px-4 py-2 gap-6 border-b border-zinc-100/50">
          <button
            onClick={() => {
              setSidebarTab("topics")
              setActiveView("chat")
            }}
            className={cn(
              "text-sm font-medium pb-2 border-b-2 transition-all",
              sidebarTab === "topics" ? "text-zinc-900 border-[#00B894]" : "text-zinc-400 border-transparent hover:text-zinc-600"
            )}
          >
            Topics
          </button>
          <button
            onClick={() => {
               setSidebarTab("dashboard")
               setActiveView("dashboard")
            }}
            className={cn(
              "text-sm font-medium pb-2 border-b-2 transition-all",
              sidebarTab === "dashboard" ? "text-zinc-900 border-[#00B894]" : "text-zinc-400 border-transparent hover:text-zinc-600"
            )}
          >
            Dashboard
          </button>
          <button
            onClick={() => {
               setSidebarTab("settings")
               setActiveView("settings")
            }}
            className={cn(
              "text-sm font-medium pb-2 border-b-2 transition-all",
              sidebarTab === "settings" ? "text-zinc-900 border-[#00B894]" : "text-zinc-400 border-transparent hover:text-zinc-600"
            )}
          >
            Settings
          </button>
        </div>

        {/* Sidebar Content */}
        <div className="flex-1 px-3 py-4 overflow-y-auto">
          
          {sidebarTab === "topics" && (
             <div className="space-y-2">
               <div className="mb-4 px-1">
                 <HaloButton onClick={createNewConversation} icon={<Plus className="w-4 h-4" />}>
                   New chat
                 </HaloButton>
               </div>
               {conversations
                  .slice()
                  .sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime())
                  .map((c) => (
                 <HistoryItem
                    key={c.id}
                    title={c.title}
                    isActive={currentConversationId === c.id}
                    onClick={() => {
                      switchConversation(c.id)
                      setActiveView("chat")
                    }}
                    onDelete={(e) => deleteConversation(c.id, e)}
                    preview={c.messages[c.messages.length - 1]?.content || "No messages"}
                    date={c.updatedAt.toLocaleString("zh-CN", { month: "numeric", day: "numeric" })}
                 />
               ))}
             </div>
          )}
        </div>
        

      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 bg-white relative overflow-hidden">
        {/* Header */}
        <header className="h-14 flex items-center justify-between px-6 border-b border-zinc-50 bg-white/80 backdrop-blur-sm z-10">

           <div className="flex items-center gap-2">
              <Button variant="ghost" size="icon" className="h-8 w-8 text-zinc-400 hover:text-zinc-600" onClick={() => {
                setActiveView("dashboard")
                setSidebarTab("dashboard")
              }}><Activity className="w-4 h-4" /></Button>
              <Button variant="ghost" size="icon" className="h-8 w-8 text-zinc-400 hover:text-zinc-600" onClick={() => {
                setActiveView("settings")
                setSidebarTab("settings")
              }}><Settings className="w-4 h-4" /></Button>
           </div>

           <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-emerald-600 bg-emerald-50 border border-emerald-100 rounded-full px-2.5 py-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                Demo Mode · Local Simulation Engine
              </span>
           </div>
        </header>

        {activeView === "chat" ? (
          <>
            <div className="flex-1 px-8 py-6 overflow-y-auto scroll-smooth" ref={scrollRef}>
               <div className="max-w-3xl mx-auto space-y-10 pb-32">
                  {messages.map((message) => (
                    <div key={message.id} className="group">
                       {message.role === "user" ? (
                         <div className="flex justify-end">
                           <div className="bg-zinc-100 text-zinc-800 px-5 py-3 rounded-2xl rounded-tr-sm max-w-[80%] text-base leading-7 shadow-sm">
                             {message.content}
                           </div>
                         </div>
                       ) : (
                         <div className="flex gap-4">
                            <div className="flex-1 space-y-4 overflow-hidden">
                              {/* Message Content */}
                              <div className="prose prose-zinc max-w-none text-zinc-700 leading-7">
                                {/* Reasoning / processing log area (shown for assistant messages with a timeline) */}
                                {message.metadata?.timeline && message.metadata.timeline.length > 0 && (
                                  <div className="mb-4 text-xs font-mono bg-slate-50 rounded-lg border border-slate-100 p-3 space-y-2 not-prose">
                                    <div className="flex items-center text-slate-500 font-medium mb-2 pb-2 border-b border-slate-100">
                                      <Sparkles className="w-3 h-3 mr-2 text-violet-500" />
                                      <span>Agent reasoning process</span>
                                    </div>
                                    {message.metadata.timeline.map((log, idx) => (
                                      <div key={idx} className="flex items-start gap-2 animate-in fade-in slide-in-from-top-1 duration-300">
                                        <div className="mt-1.5 w-1 h-1 rounded-full bg-slate-300 shrink-0" />
                                        <div className="flex flex-col">
                                          <span className="text-slate-600 leading-relaxed">{log.message}</span>
                                          {log.detail && <span className="text-slate-400 text-[10px] mt-0.5">{log.detail}</span>}
                                        </div>
                                      </div>
                                    ))}
                                    {/* show animated text while loading */}
                                    {isLoading && message.content === "🤔 Agent is thinking..." && (
                                      <div className="flex items-center gap-2 pl-3 text-violet-500 pt-1 animate-pulse">
                                        <Loader2 className="w-3 h-3 animate-spin" />
                                        <span>Running...</span>
                                      </div>
                                    )}
                                  </div>
                                )}
                                <ReactMarkdown>{message.content}</ReactMarkdown>
                              </div>

                              {/* Attachments Display */}
                              {message.attachments && message.attachments.length > 0 && (
                                <div className="flex flex-wrap gap-2">
                                  {message.attachments.map((att, idx) => (
                                     <div key={idx} className="flex items-center gap-2 bg-zinc-50 border border-zinc-100 rounded-lg px-3 py-2 text-sm text-zinc-600">
                                        {att.type === 'image' ? <ImageIcon className="w-4 h-4" /> : <Paperclip className="w-4 h-4" />}
                                        <span className="truncate max-w-[200px]">{att.name}</span>
                                     </div>
                                  ))}
                                </div>
                              )}
                              
                              {/* Metadata / Example Card */}
                              {message.metadata && (
                                 <div className="flex flex-col gap-2 mt-2">
                                    <div className="flex flex-wrap gap-2">
                                        {message.metadata.thinking && message.metadata.thinking.length > 0 && (
                                            <div className="bg-blue-50 text-blue-700 text-xs px-2 py-1 rounded-md">
                                                Thinking...
                                            </div>
                                        )}
                                        {message.metadata.retrievalTime && (
                                            <div className="text-xs text-zinc-400">Retrieval: {message.metadata.retrievalTime.toFixed(2)}s</div>
                                        )}
                                        {message.metadata.generationTime && (
                                            <div className="text-xs text-zinc-400">Generation: {message.metadata.generationTime.toFixed(2)}s</div>
                                        )}
                                    </div>
                                    {/* If there are sources, show them */}
                                    {message.metadata.sources && message.metadata.sources.length > 0 && (
                                        <div className="text-xs text-zinc-500 bg-zinc-50 p-2 rounded-lg border border-zinc-100">
                                            Sources: {message.metadata.sources.join(", ")}
                                        </div>
                                    )}
                                 </div>
                              )}
                            </div>
                         </div>
                       )}
                    </div>
                  ))}
               </div>
            </div>

            {/* Input Area - Floating Bottom */}
            <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-white via-white to-transparent z-20">
              <div className="max-w-3xl mx-auto">
                <div className="bg-white rounded-[2rem] border border-zinc-200 shadow-[0_8px_40px_-12px_rgba(0,0,0,0.1)] overflow-hidden transition-all focus-within:ring-1 focus-within:ring-primary/20 focus-within:border-primary/50">


                  {/* Textarea */}
                  <div className="px-4 py-2">
                    <Input
                       className="border-0 shadow-none focus-visible:ring-0 p-0 text-base min-h-[40px] resize-none bg-transparent placeholder:text-zinc-300"
                       placeholder="Type a message here, press Enter to send..."
                       value={inputValue}
                       onChange={(e) => setInputValue(e.target.value)}
                       onKeyDown={(e) => {
                         if (e.key === "Enter" && !e.shiftKey) {
                           e.preventDefault()
                           handleSend()
                         }
                       }}
                    />
                  </div>

                  {/* Bottom Toolbar */}
                  <div className="flex items-center justify-between px-3 py-2 bg-white">
                     <div className="flex items-center gap-1">
                        <Button variant="ghost" size="icon" className="h-9 w-9 text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100 rounded-full" onClick={() => handleFileSelect('file')}><Plus className="w-5 h-5" /></Button>
                        <Button variant="ghost" size="icon" className="h-9 w-9 text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100 rounded-full" onClick={() => handleFileSelect('image')}><ImageIcon className="w-5 h-5" /></Button>
                        <Button variant="ghost" size="icon" className="h-9 w-9 text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100 rounded-full"><Globe className="w-5 h-5" /></Button>
                     </div>
                     <div className="flex items-center gap-4">
                        <div className="text-[10px] text-zinc-300 font-mono">= {chainLength}/10 ↑ {conversations.length}</div>
                        <Button 
                          onClick={handleSend} 
                          disabled={isLoading || (!inputValue.trim() && attachments.length === 0)}
                          className={cn("rounded-full w-9 h-9 p-0 flex items-center justify-center transition-all", (inputValue.trim() || attachments.length > 0) ? "bg-black text-white shadow-md hover:bg-zinc-800" : "bg-zinc-100 text-zinc-300")}
                        >
                           {isLoading ? <RotateCcw className="w-5 h-5 animate-spin" /> : <ChevronRight className="w-5 h-5" />}
                        </Button>
                     </div>
                  </div>
                </div>
                
                {/* Attachments Preview */}
                {attachments.length > 0 && (
                  <div className="mt-2 flex gap-2 overflow-x-auto pb-2">
                    {attachments.map((att, i) => (
                      <div key={i} className="relative bg-white border border-zinc-200 rounded-lg p-2 flex items-center gap-2 shadow-sm min-w-[120px]">
                         <span className="text-xs truncate max-w-[100px]">{att.name}</span>
                         <button onClick={() => removeAttachment(i)} className="absolute -top-1 -right-1 bg-red-500 text-white rounded-full p-0.5"><X className="w-3 h-3" /></button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
            
            {/* Hidden inputs */}
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              onChange={(e) => handleFileChange(e, "file")}
            />
            <input
              type="file"
              ref={imageInputRef}
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFileChange(e, "image")}
            />
          </>
        ) : activeView === "settings" ? (
          <div className="flex-1 px-6 py-8 overflow-y-auto">
            <div className="max-w-3xl mx-auto">
              <h2 className="text-3xl font-semibold mb-8 text-balance text-zinc-800">Settings</h2>
              <div className="space-y-6">
                <div className="bg-white border border-zinc-200 rounded-2xl p-6 shadow-sm">
                  <h3 className="font-semibold text-lg mb-5 flex items-center gap-2 text-zinc-800">
                    <div className="w-2 h-2 rounded-full bg-primary"></div>
                    Model settings
                  </h3>
                  <div className="space-y-5">
                    <div>
                      <label className="text-sm font-medium text-zinc-700 mb-2 block">Reasoning chain length</label>
                      <RadioGroup
                        value={chainLevel}
                        onValueChange={(v) => {
                          setChainLevel(v)
                          setChainLength(v === "basic" ? 2 : v === "medium" ? 6 : 10)
                        }}
                        className="grid grid-cols-3 gap-2"
                      >
                        <label className="flex items-center gap-2 rounded-xl border border-zinc-200 px-3 py-2 cursor-pointer hover:bg-zinc-50 transition-colors [&:has([data-state=checked])]:border-primary [&:has([data-state=checked])]:bg-primary/5">
                          <RadioGroupItem value="basic" />
                          <span className="text-sm">Basic</span>
                        </label>
                        <label className="flex items-center gap-2 rounded-xl border border-zinc-200 px-3 py-2 cursor-pointer hover:bg-zinc-50 transition-colors [&:has([data-state=checked])]:border-primary [&:has([data-state=checked])]:bg-primary/5">
                          <RadioGroupItem value="medium" />
                          <span className="text-sm">Medium</span>
                        </label>
                        <label className="flex items-center gap-2 rounded-xl border border-zinc-200 px-3 py-2 cursor-pointer hover:bg-zinc-50 transition-colors [&:has([data-state=checked])]:border-primary [&:has([data-state=checked])]:bg-primary/5">
                          <RadioGroupItem value="advanced" />
                          <span className="text-sm">Advanced</span>
                        </label>
                      </RadioGroup>
                      <div className="text-xs text-zinc-400 mt-2">
                        {chainLevel === "basic" && "0–2 steps"}
                        {chainLevel === "medium" && "4–6 steps"}
                        {chainLevel === "advanced" && "10+ steps"}
                      </div>
                    </div>
                    {/* More settings can be added here if needed, keeping it simple for now */}
                    <div>
                       <div className="flex items-center justify-between mb-2">
                        <label className="text-sm font-medium text-zinc-700">Temperature</label>
                        <span className="text-xs text-zinc-500">{temperature.toFixed(1)}</span>
                      </div>
                      <Slider
                        value={[temperature]}
                        min={0}
                        max={1}
                        step={0.1}
                        onValueChange={([v]) => setTemperature(v)}
                        className="py-2"
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          // Dashboard View
          <div className="flex-1 px-6 py-8 overflow-y-auto bg-zinc-50/50">
             <div className="max-w-5xl mx-auto">
               <div className="flex items-center justify-between mb-8">
                 <h2 className="text-2xl font-semibold text-zinc-800">Analytics dashboard</h2>
                 <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm" onClick={saveEfficiencyBaseline} className="bg-white">Save baseline</Button>
                 </div>
               </div>
               
               <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                  <div className="bg-white p-6 rounded-2xl border border-zinc-100 shadow-sm">
                     <div className="text-sm text-zinc-500 mb-1">Total messages</div>
                     <div className="text-3xl font-bold text-zinc-900">{analytics.totalMessages}</div>
                  </div>
                  <div className="bg-white p-6 rounded-2xl border border-zinc-100 shadow-sm">
                     <div className="text-sm text-zinc-500 mb-1">Assistant replies</div>
                     <div className="text-3xl font-bold text-zinc-900">{analytics.assistantMessages}</div>
                  </div>
                  <div className="bg-white p-6 rounded-2xl border border-zinc-100 shadow-sm">
                     <div className="text-sm text-zinc-500 mb-1">Avg generation time</div>
                     <div className="text-3xl font-bold text-zinc-900">{analytics.avgGeneration.toFixed(2)}s</div>
                  </div>
               </div>

               <div className="bg-white p-6 rounded-2xl border border-zinc-100 shadow-sm mb-6">
                  <h3 className="font-medium text-zinc-800 mb-6">Latency trend</h3>
                  <div className="h-[300px] w-full">
                     {efficiencySeries.length > 0 ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={efficiencySeries}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                          <XAxis dataKey="index" tick={{ fontSize: 12, fill: '#a1a1aa' }} axisLine={false} tickLine={false} />
                          <YAxis unit="%" tick={{ fontSize: 12, fill: '#a1a1aa' }} axisLine={false} tickLine={false} />
                          <Tooltip 
                            contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                          />
                          <Legend />
                          <Line type="monotone" dataKey="retrievalGain" name="Retrieval gain" stroke="#10b981" strokeWidth={3} dot={false} />
                          <Line type="monotone" dataKey="generationGain" name="Generation gain" stroke="#3b82f6" strokeWidth={3} dot={false} />
                        </LineChart>
                      </ResponsiveContainer>
                     ) : (
                       <div className="flex items-center justify-center h-full text-zinc-400 text-sm">Not enough data yet</div>
                     )}
                  </div>
               </div>
             </div>
          </div>
        )}
      </main>
    </div>
  )
}
