// src/hooks/useGateWebSocket.js
//
// Custom hook yang menangani koneksi WebSocket ke backend Alip.
// Dipakai di DisplayGatePage (tampilan mahasiswa) DAN DashboardPage (operator).
// Keduanya connect ke endpoint yang sama → kamera yang sama terlihat di 2 halaman.

import { useEffect, useRef, useState, useCallback } from 'react'

// Ganti baris WebSocket kamu dengan ini
const ws = new WebSocket('ws://127.0.0.1:8000/ws/detect');

export const STATUS = {
  IDLE:     'idle',
  SCANNING: 'scanning',
  GRANTED:  'granted',
  DENIED:   'denied',
  SPOOF:    'spoof',
  OFFLINE:  'offline',
}

export function useGateWebSocket(canvasRef) {
  const wsRef        = useRef(null)
  const reconnectRef = useRef(null)
  const resetRef     = useRef(null)

  const [status,      setStatus]      = useState(STATUS.OFFLINE)
  const [userData,    setUserData]    = useState(null)   // { nama, nim, program_studi }
  const [boundingBox, setBoundingBox] = useState(null)   // { x1,y1,x2,y2,confidence,frameWidth,frameHeight }
  const [faceCount,   setFaceCount]   = useState(0)
  const [wsOnline,    setWsOnline]    = useState(false)
  // Log akses real-time — otomatis diisi setiap ada event granted/denied/spoof
  const [accessLogs,  setAccessLogs]  = useState([])

  // Reset ke IDLE 4 detik setelah GRANTED / DENIED / SPOOF
  const scheduleReset = useCallback(() => {
    if (resetRef.current) clearTimeout(resetRef.current)
    resetRef.current = setTimeout(() => {
      setStatus((prev) =>
        [STATUS.GRANTED, STATUS.DENIED, STATUS.SPOOF].includes(prev)
          ? STATUS.IDLE
          : prev
      )
      setUserData(null)
      setBoundingBox(null)
      setFaceCount(0)
    }, 4000)
  }, [])

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setWsOnline(true)
      setStatus(STATUS.IDLE)
      if (reconnectRef.current) clearTimeout(reconnectRef.current)
    }

    // ─── Format JSON dari backend Alip ───────────────────────────────────
    // {
    //   "image"      : "data:image/jpeg;base64,...",
    //   "status"     : "idle|scanning|granted|denied|spoof",
    //   "face_count" : 1,
    //   "faces"      : [{ x1,y1,x2,y2, confidence, frameWidth, frameHeight }],
    //   "user"       : { nama, nim, program_studi }   ← hanya saat granted
    // }
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)

        // 1. Render frame ke canvas (kalau canvasRef diberikan)
        if (msg.image && canvasRef?.current) {
          const img = new Image()
          img.onload = () => {
            const canvas = canvasRef.current
            if (!canvas) return
            canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height)
          }
          img.src = msg.image
        }

        // 2. Face count
        if (typeof msg.face_count === 'number') setFaceCount(msg.face_count)

        // 3. Bounding box
        setBoundingBox(
          Array.isArray(msg.faces) && msg.faces.length > 0 ? msg.faces[0] : null
        )

        // 4. Status & log otomatis
        const s = (msg.status || '').toLowerCase()

        if (s === 'granted') {
          setUserData(msg.user || null)
          setStatus(STATUS.GRANTED)
          scheduleReset()
          // Tambah ke log real-time
          if (msg.user) {
            setAccessLogs((prev) => [
              {
                id:     Date.now(),
                nama:   msg.user.nama   || 'Unknown',
                nim:    msg.user.nim    || '-',
                waktu:  new Date().toLocaleTimeString('id-ID'),
                gate:   'Gate 4 Masuk',
                status: 'GRANTED',
              },
              ...prev,
            ])
          }
        } else if (s === 'denied') {
          setUserData(null)
          setStatus(STATUS.DENIED)
          scheduleReset()
          setAccessLogs((prev) => [
            { id: Date.now(), nama: 'Unknown', nim: '-', waktu: new Date().toLocaleTimeString('id-ID'), gate: 'Gate 4 Masuk', status: 'DENIED' },
            ...prev,
          ])
        } else if (s === 'spoof') {
          setUserData(null)
          setStatus(STATUS.SPOOF)
          scheduleReset()
          setAccessLogs((prev) => [
            { id: Date.now(), nama: 'SPOOFING', nim: '-', waktu: new Date().toLocaleTimeString('id-ID'), gate: 'Gate 4 Masuk', status: 'DENIED' },
            ...prev,
          ])
        } else if (s === 'scanning') {
          setStatus((prev) =>
            [STATUS.GRANTED, STATUS.DENIED, STATUS.SPOOF].includes(prev) ? prev : STATUS.SCANNING
          )
        } else {
          setStatus((prev) =>
            [STATUS.GRANTED, STATUS.DENIED, STATUS.SPOOF].includes(prev) ? prev : STATUS.IDLE
          )
        }
      } catch {
        // abaikan pesan non-JSON
      }
    }

    ws.onclose = () => {
      setWsOnline(false)
      setStatus(STATUS.OFFLINE)
      reconnectRef.current = setTimeout(connect, 3000)
    }

    ws.onerror = () => ws.close()
  }, [canvasRef, scheduleReset])

  useEffect(() => {
    connect()
    return () => {
      wsRef.current?.close()
      clearTimeout(reconnectRef.current)
      clearTimeout(resetRef.current)
    }
  }, [connect])

  return {
    status,
    userData,
    boundingBox,
    faceCount,
    wsOnline,
    accessLogs,
    setAccessLogs,
    reconnect: connect,
  }
}