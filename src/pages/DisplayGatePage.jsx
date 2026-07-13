import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import logo from '../assets/logo-telkom.png'

const RECOGNITION_THRESHOLD = 0.7

function DisplayGatePage() {
  const navigate = useNavigate()

  // ─── Refs ──────────────────────────────────────────────────────
  const frameCountRef = useRef(0)
  const lastFpsUpdateRef = useRef(Date.now())
  const overlayTimeoutRef = useRef(null)
  const statusPrevRef = useRef(null)
  const wsRef = useRef(null)
  const nimCooldownRef = useRef({})
  const deniedTimeoutRef = useRef(null)
  const COOLDOWN_MS = 10000

  // ─── State ────────────────────────────────────────────────────
  const [isAuthorized, setIsAuthorized] = useState(false)
  const [isChecking, setIsChecking] = useState(true)
  const [frame, setFrame] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [currentDetection, setCurrentDetection] = useState(null)
  const [showOverlay, setShowOverlay] = useState(false)
  const [stats, setStats] = useState({ granted: 0, denied: 0, total: 0 })
  const [recentLogs, setRecentLogs] = useState([])
  const [systemStatus, setSystemStatus] = useState('menunggu')
  const [fps, setFps] = useState(0)
  const [antispofEnabled, setAntispofEnabled] = useState(true)
  const [livenessPassed, setLivenessPassed] = useState(false)
  const [livenessChallenge, setLivenessChallenge] = useState('')
  const [livenessEar, setLivenessEar] = useState(0)
  const [livenessYaw, setLivenessYaw] = useState(0)
  const [livenessSmile, setLivenessSmile] = useState(0)
  const [realtimeConfidence, setRealtimeConfidence] = useState(null)
  const [lastGrantedTime, setLastGrantedTime] = useState(0)

  // ─── Auth Check ───────────────────────────────────────────────
  useEffect(() => {
    const isLoggedIn = localStorage.getItem('isLoggedIn')
    if (isLoggedIn !== 'true') {
      navigate('/')
    } else {
      setIsAuthorized(true)
      setIsChecking(false)
    }
  }, [navigate])

  // ─── WebSocket ────────────────────────────────────────────────
  const sendAntispofStatus = (status) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ antispof: status }))
    }
  }

  useEffect(() => {
    if (!isAuthorized) return

    const ws = new WebSocket('ws://localhost:8000/ws/detect')
    wsRef.current = ws

    ws.onopen = () => {
      setIsConnected(true)
      setSystemStatus('terhubung')
      sendAntispofStatus(antispofEnabled)
    }

    ws.onclose = () => {
      setIsConnected(false)
      setSystemStatus('putus')
      setTimeout(() => {
        if (wsRef.current?.readyState !== WebSocket.OPEN) {
          const newWs = new WebSocket('ws://localhost:8000/ws/detect')
          wsRef.current = newWs
          newWs.onopen = ws.onopen
          newWs.onclose = ws.onclose
          newWs.onerror = ws.onerror
          newWs.onmessage = ws.onmessage
        }
      }, 3000)
    }

    ws.onerror = () => setSystemStatus('error')

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.image) setFrame(data.image)

        // FPS
        frameCountRef.current++
        const now = Date.now()
        if (now - lastFpsUpdateRef.current >= 1000) {
          setFps(frameCountRef.current)
          frameCountRef.current = 0
          lastFpsUpdateRef.current = now
        }

        // Liveness
        if (data.liveness) {
          setLivenessPassed(data.liveness.passed || false)
          setLivenessChallenge(data.liveness.challenge || '')
          setLivenessEar(data.liveness.ear || 0)
          setLivenessYaw(data.liveness.yaw || 0)
          setLivenessSmile(data.liveness.smile || 0)
        }

        // Confidence
        if (data.liveness?.passed && data.faces?.length > 0) {
          setRealtimeConfidence(data.faces[0].confidence ?? null)
        } else if (!data.liveness?.passed) {
          setRealtimeConfidence(null)
        }

        const currentStatus = data.status
        const prevStatus = statusPrevRef.current

        if (currentStatus !== prevStatus) {
          statusPrevRef.current = currentStatus

          if (currentStatus === 'scanning' || currentStatus === 'idle') {
            if (deniedTimeoutRef.current) {
              clearTimeout(deniedTimeoutRef.current)
              deniedTimeoutRef.current = null
            }
            if (overlayTimeoutRef.current) clearTimeout(overlayTimeoutRef.current)
            setShowOverlay(false)
          }

          if (currentStatus === 'granted') {
            // Catat waktu granted
            setLastGrantedTime(Date.now())

            if (deniedTimeoutRef.current) {
              clearTimeout(deniedTimeoutRef.current)
              deniedTimeoutRef.current = null
            }

            const detection = {
              nama: data.user?.nama || 'Unknown',
              nim: data.user?.nim || '-',
              status: 'granted',
              confidence: data.user?.confidence || 0,
              waktu: new Date().toLocaleTimeString('id-ID')
            }
            setCurrentDetection(detection)
            setSystemStatus('granted')
            if (overlayTimeoutRef.current) clearTimeout(overlayTimeoutRef.current)
            setShowOverlay(true)
            overlayTimeoutRef.current = setTimeout(() => setShowOverlay(false), 2000)
          }

          if (currentStatus === 'denied') {
            if (deniedTimeoutRef.current) {
              clearTimeout(deniedTimeoutRef.current)
              deniedTimeoutRef.current = null
            }

            deniedTimeoutRef.current = setTimeout(() => {
              // CEK: Apakah ada GRANTED dalam 1.5 detik terakhir?
              const now3 = Date.now()
              if (now3 - lastGrantedTime < 1500) {
                // Ada granted baru, batalkan denied!
                console.log('⏭️ Denied dibatalkan karena ada granted terbaru')
                deniedTimeoutRef.current = null
                return
              }

              // OVERLAY
              const detection = {
                nama: 'Unknown',
                nim: '-',
                status: 'denied',
                confidence: 0,
                waktu: new Date().toLocaleTimeString('id-ID')
              }
              setCurrentDetection(detection)
              setSystemStatus('denied')
              if (overlayTimeoutRef.current) clearTimeout(overlayTimeoutRef.current)
              setShowOverlay(true)
              overlayTimeoutRef.current = setTimeout(() => setShowOverlay(false), 2000)

              // LOG & STATS (hanya untuk denied)
              const now2 = Date.now()
              const cooldownKey = 'DENIED_GLOBAL'
              const lastTime = nimCooldownRef.current[cooldownKey] || 0

              if (now2 - lastTime >= COOLDOWN_MS) {
                nimCooldownRef.current[cooldownKey] = now2
                const newLog = {
                  id: now2,
                  nama: 'Unknown',
                  nim: '-',
                  status: 'DENIED',
                  waktu: new Date().toLocaleTimeString('id-ID'),
                  confidence: '0'
                }
                setRecentLogs(prev => [newLog, ...prev].slice(0, 10))
                setStats(prev => ({
                  granted: prev.granted,
                  denied: prev.denied + 1,
                  total: prev.total + 1
                }))
              }

              deniedTimeoutRef.current = null
            }, 400)
          }
        }

        // LOGS & STATS (hanya untuk GRANTED)
        if (data.faces?.length > 0 && currentStatus === 'granted') {
          const cooldownKey = data.user?.nim || 'UNKNOWN'
          const now2 = Date.now()
          const lastTime = nimCooldownRef.current[cooldownKey] || 0

          if (now2 - lastTime >= COOLDOWN_MS) {
            nimCooldownRef.current[cooldownKey] = now2
            const newLog = {
              id: now2,
              nama: data.user?.nama || 'Unknown',
              nim: data.user?.nim || '-',
              status: 'GRANTED',
              waktu: new Date().toLocaleTimeString('id-ID'),
              confidence: ((data.user?.confidence || 0) * 100).toFixed(1)
            }
            setRecentLogs(prev => [newLog, ...prev].slice(0, 10))
            setStats(prev => ({
              granted: prev.granted + 1,
              denied: prev.denied,
              total: prev.total + 1
            }))
          }
        }
      } catch (err) {
        console.error('WS parse error:', err)
      }
    }

    return () => {
      if (overlayTimeoutRef.current) clearTimeout(overlayTimeoutRef.current)
      if (ws.readyState === WebSocket.OPEN) ws.close()
    }
  }, [isAuthorized])

  useEffect(() => {
    sendAntispofStatus(antispofEnabled)
  }, [antispofEnabled])

  // ─── Helpers ──────────────────────────────────────────────────
  const getStatusColor = () => {
    const colors = {
      granted: 'text-green-400',
      denied: 'text-red-400',
      terhubung: 'text-emerald-400',
      putus: 'text-red-400',
      error: 'text-amber-400'
    }
    return colors[systemStatus] || 'text-slate-400'
  }

  const getStatusBadge = () => {
    const badges = {
      granted: { bg: 'bg-green-900/60', border: 'border-green-500/40', text: 'ACCESS GRANTED', icon: '✓' },
      denied: { bg: 'bg-red-900/60', border: 'border-red-500/40', text: 'ACCESS DENIED', icon: '✗' },
      terhubung: { bg: 'bg-emerald-900/60', border: 'border-emerald-500/40', text: 'SYSTEM READY', icon: '●' },
      putus: { bg: 'bg-red-900/60', border: 'border-red-500/40', text: 'CONNECTION LOST', icon: '⚠' },
      error: { bg: 'bg-amber-900/60', border: 'border-amber-500/40', text: 'SYSTEM ERROR', icon: '!' },
      scanning: { bg: 'bg-blue-900/60', border: 'border-blue-500/40', text: 'SCANNING...', icon: '🔍' }
    }
    return badges[systemStatus] || { bg: 'bg-gray-500/60', border: 'border-gray-500/40', text: 'AWAITING DETECTION', icon: '○' }
  }

  const statusBadge = getStatusBadge()

  const getChallengeInstruction = () => {
    const instructions = {
      BLINK: 'Blink Your Eyes',
      HEAD: 'Turn Your Head',
      SMILE: 'Smile'
    }
    return instructions[livenessChallenge] || ''
  }

  const getConfidenceBar = (confidence) => {
    const pct = Math.min((confidence / RECOGNITION_THRESHOLD) * 100, 100)
    let color
    if (pct < 60) {
      color = {
        bar: 'bg-red-500',
        text: 'text-red-400',
        label: 'bg-red-500/20 border-red-500/30',
        track: 'bg-red-500/10'
      }
    } else if (pct < 90) {
      color = {
        bar: 'bg-yellow-400',
        text: 'text-yellow-400',
        label: 'bg-yellow-500/20 border-yellow-500/30',
        track: 'bg-yellow-500/10'
      }
    } else {
      color = {
        bar: 'bg-green-500',
        text: 'text-green-400',
        label: 'bg-green-500/20 border-green-500/30',
        track: 'bg-green-500/10'
      }
    }
    return { pct, color }
  }

  // ─── Render ───────────────────────────────────────────────────
  if (isChecking) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-black via-red-950 to-black flex items-center justify-center">
        <p className="text-white text-lg font-mono animate-pulse">Authenticating...</p>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-red-950 to-black text-white">

      {/* NAVBAR */}
      <nav className="bg-white backdrop-blur px-6 py-3.5 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <img src={logo} alt="logo" className="w-8 h-8 object-contain" />
          <div>
            <p className="text-black font-bold text-sm leading-tight">Face Recognition Gate</p>
            <p className="text-gray-900 text-[11px]">Display Gate 4</p>
          </div>
        </div>
        <button
          onClick={() => {
            localStorage.removeItem('isLoggedIn')
            navigate('/')
          }}
          className="text-gray-700 hover:text-red-500 text-sm font-semibold transition-all duration-200 hover:scale-105"
        >
          Logout ⮞
        </button>
      </nav>

      {/* MAIN CONTENT */}
      <div className="p-6 max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* LEFT: VIDEO FEED + STATISTICS */}
          <div className="lg:col-span-2 space-y-4">

            {/* Video Feed */}
            <div className="bg-gray-500/80 backdrop-blur-sm rounded-2xl border border-gray-800 overflow-hidden shadow-xl">
              <div className="px-5 py-3 bg-gray-800/50 border-b border-gray-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full animate-pulse ${isConnected ? 'bg-green-400' : 'bg-red-400'}`} />
                  <span className={`text-xs font-mono ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
                    {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
                  </span>
                </div>
                <span className="text-gray-400 text-xs font-mono">{fps} FPS</span>
              </div>

              <div className="relative aspect-video bg-black">
                {frame ? (
                  <img src={frame} alt="Live Feed" className="w-full h-full object-cover" />
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <div className="w-16 h-16 bg-gray-800 rounded-2xl flex items-center justify-center mb-3">
                      <svg className="w-8 h-8 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <p className="text-gray-500 text-sm">Awaiting Camera</p>
                    <p className="text-gray-500 text-xs mt-1">{isConnected ? 'Connected' : 'Connecting...'}</p>
                  </div>
                )}

                {/* Liveness Challenge */}
                {antispofEnabled && !livenessPassed && livenessChallenge && (
                  <div className="absolute bottom-4 left-4 bg-yellow-600/80 backdrop-blur-sm rounded-full px-4 py-2 shadow-lg">
                    <span className="text-white text-[18px] font-bold" style={{ WebkitTextStroke: '0.5px black' }}>
                      {getChallengeInstruction()}
                    </span>
                  </div>
                )}

                {/* Overlay */}
                {showOverlay && currentDetection && (systemStatus === 'granted' || systemStatus === 'denied') && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-in fade-in duration-300">
                    <div className={`text-center p-8 rounded-2xl border-2 shadow-2xl ${systemStatus === 'granted'
                        ? 'bg-green-900/90 border-green-500'
                        : 'bg-red-900/90 border-red-500'
                    }`}>
                      <div className={`text-6xl mb-3 ${systemStatus === 'granted' ? 'text-green-400' : 'text-red-400'}`}>
                        {systemStatus === 'granted' ? '✓' : '✗'}
                      </div>
                      <h2 className={`font-bold text-2xl tracking-wider mb-2 ${systemStatus === 'granted' ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {systemStatus === 'granted' ? 'ACCESS GRANTED' : 'ACCESS DENIED'}
                      </h2>
                      {systemStatus === 'granted' && (
                        <>
                          <p className="text-white text-xl font-semibold">{currentDetection.nama}</p>
                          <p className="text-gray-300 text-sm">{currentDetection.nim}</p>
                          <p className="text-gray-400 text-xs mt-2">
                            Similarity: {(Number(currentDetection.confidence) * 100).toFixed(2)}%
                          </p>
                        </>
                      )}
                      {systemStatus === 'denied' && (
                        <p className="text-gray-300 text-sm">Face not recognized in the database</p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Statistics */}
            <div className="bg-gray-500/80 backdrop-blur-sm rounded-2xl border border-gray-800 p-5 shadow-xl">
              <p className="text-white text-sm font-bold uppercase tracking-wider mb-4">Today's Statistics</p>
              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: 'GRANTED', value: stats.granted, color: 'text-green-400' },
                  { label: 'DENIED', value: stats.denied, color: 'text-red-400' },
                  { label: 'TOTAL', value: stats.total, color: 'text-red-400' }
                ].map((item) => (
                  <div key={item.label} className="text-center p-3 bg-black/30 rounded-xl border border-gray-700/50">
                    <p className={`text-2xl font-bold ${item.color}`}>{item.value}</p>
                    <p className="text-gray-300 text-[12px] font-medium mt-1">{item.label}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Mobile Status */}
            <div className="lg:hidden bg-gray-500/80 rounded-2xl border border-gray-800 p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full animate-pulse ${isConnected ? 'bg-green-400' : 'bg-red-400'}`} />
                  <span className="text-gray-300 text-sm">Connection Status</span>
                </div>
                <span className={`text-xs font-mono ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
          </div>

          {/* RIGHT: System Status + Recent Activities */}
          <div className="space-y-5">

            {/* System Status */}
            <div className="bg-gray-500/80 backdrop-blur-sm rounded-2xl border border-gray-800 p-5 shadow-xl">
              <div className="flex items-center justify-between mb-4">
                <p className="text-white text-[18px] font-bold uppercase tracking-wider">System Status</p>
                <div className={`px-2 py-1 rounded-lg text-[10px] font-mono font-bold ${statusBadge.bg} ${statusBadge.border} ${getStatusColor()}`}>
                  {statusBadge.text}
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className={`w-14 h-14 rounded-2xl flex items-center justify-center text-2xl ${statusBadge.bg} ${statusBadge.border}`}>
                  {statusBadge.icon}
                </div>
                <div>
                  <p className="text-gray-100 text-[12px]">Last Detection</p>
                  {currentDetection ? (
                    <>
                      <p className="text-white font-semibold text-lg">{currentDetection.nama}</p>
                      <p className="text-white font-semibold text-sm">{currentDetection.nim}</p>
                      <p className="text-gray-300 text-xs mt-1">{currentDetection.waktu}</p>
                    </>
                  ) : (
                    <p className="text-gray-300 text-sm">No detection yet</p>
                  )}
                </div>
              </div>

              {/* Liveness Check */}
              <div className="mt-4 pt-3 border-t border-gray-700/50">
                <p className="text-white text-sm font-bold uppercase tracking-wider mb-2">Liveness Check</p>
                {!antispofEnabled ? (
                  <div className="p-2 bg-black/30 rounded-lg text-center">
                    <p className="text-gray-400 text-sm">⚙️ Anti-spoofing disabled</p>
                  </div>
                ) : (
                  <>
                    {!livenessPassed ? (
                      <p className="text-yellow-400 text-sm font-semibold">
                        Challenge: {getChallengeInstruction() || livenessChallenge || '—'}
                      </p>
                    ) : (
                      <p className="text-green-400 text-sm font-semibold">✓ Liveness verified</p>
                    )}
                    <div className="grid grid-cols-3 gap-2 mt-2">
                      {[
                        { label: 'Blink', value: livenessEar },
                        { label: 'Head', value: livenessYaw },
                        { label: 'Smile', value: livenessSmile }
                      ].map((item) => (
                        <div key={item.label} className="flex items-center bg-black/30 rounded-lg px-2 py-1.5">
                          <div className="flex flex-col w-full">
                            <span className="text-xs text-gray-400">{item.label}</span>
                            <span className="font-mono text-gray-300 text-sm">{item.value.toFixed(2)}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>

              {/* Confidence Bar */}
              {livenessPassed && realtimeConfidence !== null && (() => {
                const { pct, color } = getConfidenceBar(realtimeConfidence)
                return (
                  <div className="mt-4 pt-3 border-t border-gray-700/50">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-white text-sm font-bold uppercase tracking-wider">Facial Similarity</p>
                      <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded-lg border ${color.label} ${color.text}`}>
                        {pct.toFixed(1)}%
                      </span>
                    </div>
                    <div className={`w-full rounded-full h-3 overflow-hidden ${color.track} border border-gray-700`}>
                      <div className={`h-3 rounded-full transition-all duration-300 ${color.bar}`} style={{ width: `${pct}%` }} />
                    </div>
                    <p className="text-gray-500 text-[10px] font-mono mt-1 text-right">
                      Score: {realtimeConfidence.toFixed(4)}
                    </p>
                  </div>
                )
              })()}
            </div>

            {/* Recent Activities */}
            <div className="bg-gray-500/80 backdrop-blur-sm rounded-2xl border border-gray-800 shadow-xl overflow-hidden">
              <div className="px-5 py-3 border-b border-gray-800">
                <p className="text-white text-sm font-bold uppercase tracking-wider">Recent Activities</p>
              </div>
              <div className="max-h-64 overflow-y-auto">
                {recentLogs.length === 0 ? (
                  <div className="p-8 text-center">
                    <p className="text-gray-200 text-sm">No recent activities</p>
                  </div>
                ) : (
                  <div className="divide-y divide-gray-700/50">
                    {recentLogs.map((log) => (
                      <div key={log.id} className="px-5 py-3 hover:bg-gray-700/30 transition-colors">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-white text-[14px] font-medium">{log.nama}</p>
                            <p className="text-gray-200 text-[13px]">{log.nim}</p>
                            <p className="text-gray-300 text-[12px] font-mono">{log.waktu}</p>
                            {log.confidence && (
                              <p className="text-yellow-400 text-[11px] mt-1">Confidence: {log.confidence}%</p>
                            )}
                          </div>
                          <span className={`text-[10px] font-bold px-2 py-1 rounded-lg ${log.status === 'GRANTED'
                              ? 'bg-green-900/60 text-green-400 border border-green-500/30'
                              : 'bg-red-900/60 text-red-400 border border-red-500/30'
                            }`}>{log.status}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DisplayGatePage