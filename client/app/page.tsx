'use client'

import classNames from "classnames";
import Image from "next/image";
import { useCallback, useEffect, useRef, useState } from "react";

const TIMESLICE = 1000

export default function Home() {

  const socket = useRef<WebSocket | null>(null)

  const makeSocket = useCallback(async () => {
    if (socket.current) {
      console.log('Using existing socket')
      return socket.current
    }
    console.log('Making socket...')
    try {
      const _socket = new WebSocket("ws://localhost:8080/ws")

      _socket.addEventListener("open", () => {
        socket.current = _socket
        console.log('connection opened')
      })
      _socket.addEventListener("close", () => {
        socket.current = null
        console.log("Connection closed, reconnecting...")
        makeSocket()
      })
      // Listen for messages
      _socket.addEventListener("message", (event) => {
        console.log("Message from server ", event.data);
        // setTranscript
      });
      return _socket
    } catch {
      console.error('Failed to make socket, retrying in 5 seconds')
      await new Promise((resolve) => setTimeout(resolve, 5000))
      return makeSocket()
    }
  }, [])
  useEffect(() => {
    console.log('Making socket on mount')
    makeSocket()
    return () => {
      if (socket.current) {
        console.log('Closing socket on unmount')
        socket.current.close()
      }
    }
  }, [makeSocket])

  const [transcript, setTranscript] = useState<{ speaker: string, text: string, start_ms: number }[]>([])

  const mediaRecorder = useRef<MediaRecorder | null>(null)
  const [mediaChunks, setMediaChunks] = useState<Blob[]>([])
  const [recording, setRecording] = useState(false)
  useEffect(() => {
    if (!mediaRecorder.current) {
      navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        mediaRecorder.current = new MediaRecorder(stream, { mimeType: 'audio/webm; codecs=opus'});
  
        mediaRecorder.current.ondataavailable = async (e) => {

          if (e.data.size === 0){
            return
          }

          console.log('got data')

          const _socket = await makeSocket()
          const audioBlob = new Blob([e.data], { type: 'audio/webm; codecs=opus' });
          _socket.send(audioBlob);
        };

        mediaRecorder.current.onstart = () => {
          setRecording(true)
        }

        mediaRecorder.current.onstop = (e) => {
          setRecording(false)
          console.log("data available after MediaRecorder.stop() called.");
        };
      })
    }
  }, [makeSocket])

  // console.log(mediaChunks)
  // useEffect(() => {

  // }, mediaChunks)

  

  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-8 row-start-2 items-center sm:items-start">

        <button onClick={() => {
          if (!mediaRecorder.current) {
            throw new Error("MediaRecorder is not initialized")
          }
          if (!recording) {
            mediaRecorder.current.start(TIMESLICE);
            console.log(mediaRecorder.current.state);
            console.log("recorder started");
          } else {
            mediaRecorder.current.stop();
            console.log(mediaRecorder.current.state);
            console.log("recorder stopped");
          }
        }} className={classNames(recording ? 'bg-red-500' : 'bg-neutral-200')}>{recording ? 'Stop' : 'Start'}</button>

        <Image
          className="dark:invert"
          src="/next.svg"
          alt="Next.js logo"
          width={180}
          height={38}
          priority
        />
        <ol className="list-inside list-decimal text-sm text-center sm:text-left font-[family-name:var(--font-geist-mono)]">
          <li className="mb-2">
            Get started by editing{" "}
            <code className="bg-black/[.05] dark:bg-white/[.06] px-1 py-0.5 rounded font-semibold">
              app/page.tsx
            </code>
            .
          </li>
          <li>Save and see your changes instantly.</li>
        </ol>

        <div className="flex gap-4 items-center flex-col sm:flex-row">
          <a
            className="rounded-full border border-solid border-transparent transition-colors flex items-center justify-center bg-foreground text-background gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5"
            href="https://vercel.com/new?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Image
              className="dark:invert"
              src="/vercel.svg"
              alt="Vercel logomark"
              width={20}
              height={20}
            />
            Deploy now
          </a>
          <a
            className="rounded-full border border-solid border-black/[.08] dark:border-white/[.145] transition-colors flex items-center justify-center hover:bg-[#f2f2f2] dark:hover:bg-[#1a1a1a] hover:border-transparent text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 sm:min-w-44"
            href="https://nextjs.org/docs?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            Read our docs
          </a>
        </div>
      </main>
      <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center">
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/file.svg"
            alt="File icon"
            width={16}
            height={16}
          />
          Learn
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/window.svg"
            alt="Window icon"
            width={16}
            height={16}
          />
          Examples
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://nextjs.org?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/globe.svg"
            alt="Globe icon"
            width={16}
            height={16}
          />
          Go to nextjs.org →
        </a>
      </footer>
    </div>
  );
}
